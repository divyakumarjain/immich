import {
  FaceGraphNodeKind,
  type FaceGraphEdgeDto,
  type FaceGraphNodeDto,
  type FaceGroupDto,
  type FaceGroupFaceDto,
} from '@immich/sdk';
import { forceCollide, forceLink, forceSimulation, forceX, forceY } from 'd3-force';
import { normalizeSearchString } from '$lib/utils/string-utils';

export type FaceGraphPosition = { x: number; y: number };

export type FaceGraphFilters = {
  unnamedOnly: boolean;
  showHidden: boolean;
  showUnassigned: boolean;
  minPhotos: number;
};

const MIN_RADIUS = 6;
const MAX_RADIUS = 60;
const NODE_PADDING = 3;
const LAYOUT_TICKS = 300;

/** the area of a node is proportional to the number of photos of the person */
export const nodeRadius = (assetCount: number) =>
  Math.min(MAX_RADIUS, Math.max(MIN_RADIUS, 2 * Math.sqrt(Math.max(assetCount, 0))));

export const isUnnamed = (node: FaceGraphNodeDto) => node.name === '';

export const isUnassigned = (node: FaceGraphNodeDto) => node.kind === FaceGraphNodeKind.Unassigned;

export const matchesFilters = (node: FaceGraphNodeDto, filters: FaceGraphFilters) => {
  if (isUnassigned(node) && !filters.showUnassigned) {
    return false;
  }
  if (node.isHidden && !filters.showHidden) {
    return false;
  }
  if (filters.unnamedOnly && !isUnnamed(node)) {
    return false;
  }
  return node.assetCount >= filters.minPhotos;
};

/** named people matching the search, the ones with the most photos first */
export const searchNodes = (nodes: FaceGraphNodeDto[], search: string) => {
  const query = normalizeSearchString(search);
  if (!query) {
    return [];
  }
  return nodes
    .filter((node) => normalizeSearchString(node.name).includes(query))
    .toSorted((a, b) => b.assetCount - a.assetCount);
};

/** unnamed people with the most photos first */
export const getWorkQueue = (nodes: FaceGraphNodeDto[]) =>
  nodes.filter((node) => isUnnamed(node)).toSorted((a, b) => b.assetCount - a.assetCount || a.id.localeCompare(b.id));

/** the nodes linked to the given node, most similar first */
export const getNeighbors = (id: string, edges: FaceGraphEdgeDto[]) =>
  edges
    .filter((edge) => edge.source === id || edge.target === id)
    .map((edge) => ({ id: edge.source === id ? edge.target : edge.source, distance: edge.distance }))
    .toSorted((a, b) => a.distance - b.distance);

/**
 * Spreads the nodes out so they do not overlap, while keeping linked nodes close together
 * and unrelated nodes near the position suggested by the server.
 */
export const computeLayout = (nodes: FaceGraphNodeDto[], edges: FaceGraphEdgeDto[]) => {
  const totalArea = nodes.reduce((sum, node) => sum + Math.PI * (nodeRadius(node.assetCount) + NODE_PADDING) ** 2, 0);
  const scale = Math.sqrt(totalArea) * 1.5;

  const simulationNodes = nodes.map((node) => ({
    id: node.id,
    radius: nodeRadius(node.assetCount),
    seedX: node.x * scale,
    seedY: node.y * scale,
    x: node.x * scale,
    y: node.y * scale,
  }));
  type SimulationNode = (typeof simulationNodes)[number];

  const ids = new Set(nodes.map(({ id }) => id));
  const links = edges
    .filter(({ source, target }) => ids.has(source) && ids.has(target))
    .map(({ source, target, distance }) => ({ source, target, distance }));
  type SimulationLink = { source: SimulationNode | string; target: SimulationNode | string; distance: number };

  forceSimulation(simulationNodes)
    .force(
      'link',
      forceLink<SimulationNode, SimulationLink>(links)
        .id((node) => node.id)
        .distance(
          ({ source, target, distance }) =>
            (source as SimulationNode).radius + (target as SimulationNode).radius + NODE_PADDING + distance * 100,
        )
        .strength(0.4),
    )
    .force(
      'collide',
      forceCollide<SimulationNode>((node) => node.radius + NODE_PADDING),
    )
    .force('x', forceX<SimulationNode>((node) => node.seedX).strength(0.05))
    .force('y', forceY<SimulationNode>((node) => node.seedY).strength(0.05))
    .stop()
    .tick(LAYOUT_TICKS);

  return new Map<string, FaceGraphPosition>(simulationNodes.map(({ id, x, y }) => [id, { x, y }]));
};

type FaceBox = Pick<
  FaceGroupFaceDto,
  'imageWidth' | 'imageHeight' | 'boundingBoxX1' | 'boundingBoxY1' | 'boundingBoxX2' | 'boundingBoxY2'
>;

// how much of the surroundings of a face is shown, relative to the size of the face
const FACE_CROP_PADDING = 1.5;
// faces smaller than this part of the image are too blurry in a thumbnail
const FACE_THUMBNAIL_MIN_RATIO = 0.3;

/** the square around a face, in pixels of the image the face was detected in */
export const getFaceCrop = (face: FaceBox) => {
  const width = face.boundingBoxX2 - face.boundingBoxX1;
  const height = face.boundingBoxY2 - face.boundingBoxY1;
  const size = Math.max(1, Math.min(Math.max(width, height) * FACE_CROP_PADDING, face.imageWidth, face.imageHeight));
  const centerX = (face.boundingBoxX1 + face.boundingBoxX2) / 2;
  const centerY = (face.boundingBoxY1 + face.boundingBoxY2) / 2;

  return {
    size,
    x: Math.min(Math.max(centerX - size / 2, 0), Math.max(face.imageWidth - size, 0)),
    y: Math.min(Math.max(centerY - size / 2, 0), Math.max(face.imageHeight - size, 0)),
  };
};

/** positions the whole image inside a square tile so only the face is visible */
export const getFaceCropStyle = (face: FaceBox, tileSize: number) => {
  const crop = getFaceCrop(face);
  const scale = tileSize / crop.size;
  return {
    width: face.imageWidth * scale,
    height: face.imageHeight * scale,
    left: -crop.x * scale,
    top: -crop.y * scale,
  };
};

export const isLargeFace = (face: FaceBox) =>
  getFaceCrop(face).size / Math.max(1, Math.min(face.imageWidth, face.imageHeight)) >= FACE_THUMBNAIL_MIN_RATIO;

/** a group is suspicious when it looks more like someone else than like the rest of the person */
export const looksLikeSomeoneElse = (group: Pick<FaceGroupDto, 'closestPerson' | 'distanceToMain'>) =>
  !!group.closestPerson && group.closestPerson.distance < group.distanceToMain;

/** another person with the same name, ignoring case and accents */
export const findNodeByName = (nodes: FaceGraphNodeDto[], name: string, excludeId: string) => {
  const query = normalizeSearchString(name);
  if (!query) {
    return;
  }
  return nodes.find(
    (node) => node.id !== excludeId && !isUnassigned(node) && normalizeSearchString(node.name) === query,
  );
};

/** the person the others are merged into: a named one if there is any, otherwise the one with the most photos */
export const findMergeTarget = (nodes: FaceGraphNodeDto[]) =>
  nodes
    .filter((node) => !isUnassigned(node))
    .toSorted((a, b) => Number(isUnnamed(a)) - Number(isUnnamed(b)) || b.assetCount - a.assetCount)
    .at(0);
