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
  /** `undefined` means no upper limit */
  maxPhotos?: number;
};

const MIN_RADIUS = 6;
const MAX_RADIUS = 60;
const NODE_PADDING = 3;
const LAYOUT_TICKS = 300;
// how much the layout is stirred up while a node is dragged
const DRAG_ALPHA = 0.3;

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
  return (
    node.assetCount >= filters.minPhotos && (filters.maxPhotos === undefined || node.assetCount <= filters.maxPhotos)
  );
};

/** named people matching the search, the ones with the most photos first */
export const searchNodes = (nodes: FaceGraphNodeDto[], search: string) => {
  const query = normalizeSearchString(search.trim());
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
 * Only the shown nodes take up room, see `show`.
 */
export const createLayout = (
  nodes: FaceGraphNodeDto[],
  edges: FaceGraphEdgeDto[],
  visible: FaceGraphNodeDto[] = nodes,
) => {
  type SimulationNode = {
    id: string;
    radius: number;
    /** position suggested by the server, between -1 and 1 */
    unitX: number;
    unitY: number;
    seedX: number;
    seedY: number;
    x: number;
    y: number;
    fx?: number;
    fy?: number;
  };
  type SimulationLink = { source: SimulationNode | string; target: SimulationNode | string; distance: number };

  const byId = new Map<string, SimulationNode>(
    nodes.map((node) => [
      node.id,
      { id: node.id, radius: 0, unitX: node.x, unitY: node.y, seedX: 0, seedY: 0, x: 0, y: 0 },
    ]),
  );
  let shown: SimulationNode[] = [];

  const linkForce = forceLink<SimulationNode, SimulationLink>([])
    .id((node) => node.id)
    .distance(
      ({ source, target, distance }) =>
        (source as SimulationNode).radius + (target as SimulationNode).radius + NODE_PADDING + distance * 100,
    )
    .strength(0.4);

  const simulation = forceSimulation<SimulationNode>([])
    .force('link', linkForce)
    .force(
      'collide',
      forceCollide<SimulationNode>((node) => node.radius + NODE_PADDING),
    )
    .force('x', forceX<SimulationNode>((node) => node.seedX).strength(0.05))
    .force('y', forceY<SimulationNode>((node) => node.seedY).strength(0.05))
    .stop();

  /**
   * Lays out the given nodes only: hidden nodes leave no gaps and the graph shrinks to fit the rest.
   * Returns whether anything changed.
   */
  const show = (visible: FaceGraphNodeDto[]) => {
    const next = visible.map((node) => byId.get(node.id)).filter((node) => !!node);
    const radii = visible.map((node) => nodeRadius(node.assetCount));
    const isSame =
      next.length === shown.length &&
      next.every((node, index) => node === shown[index] && node.radius === radii[index]);
    if (isSame) {
      return false;
    }

    const previous = new Set(shown);
    const totalArea = radii.reduce((sum, radius) => sum + Math.PI * (radius + NODE_PADDING) ** 2, 0);
    const scale = Math.sqrt(totalArea) * 1.5;
    for (const [index, node] of next.entries()) {
      node.radius = radii[index];
      node.seedX = node.unitX * scale;
      node.seedY = node.unitY * scale;
      if (previous.has(node)) {
        continue;
      }

      // a node that comes back starts where it belongs, not where it was when it was hidden
      node.x = node.seedX;
      node.y = node.seedY;
    }

    shown = next;
    const ids = new Set(shown.map(({ id }) => id));
    simulation.nodes(shown);
    // forces cache per-node values, so they are set up again for the new nodes
    linkForce.links(
      edges
        .filter(({ source, target }) => ids.has(source) && ids.has(target))
        .map(({ source, target, distance }) => ({ source, target, distance })),
    );
    simulation.force(
      'collide',
      forceCollide<SimulationNode>((node) => node.radius + NODE_PADDING),
    );
    simulation.force('x', forceX<SimulationNode>((node) => node.seedX).strength(0.05));
    simulation.force('y', forceY<SimulationNode>((node) => node.seedY).strength(0.05));
    simulation.alpha(1);
    return true;
  };

  show(visible);
  simulation.tick(LAYOUT_TICKS);

  return {
    getPositions: () => new Map<string, FaceGraphPosition>(shown.map(({ id, x, y }) => [id, { x, y }])),
    show,
    /** holds a node at the given position, the other nodes make room for it on the next steps */
    pin: (id: string, position: FaceGraphPosition) => {
      const node = byId.get(id);
      if (node) {
        node.fx = position.x;
        node.fy = position.y;
        simulation.alpha(Math.max(simulation.alpha(), DRAG_ALPHA));
      }
    },
    /** advances the layout, returns whether it is still moving */
    step: (ticks = 1) => {
      simulation.tick(ticks);
      return simulation.alpha() > simulation.alphaMin();
    },
  };
};

export type FaceGraphLayout = ReturnType<typeof createLayout>;

export const computeLayout = (nodes: FaceGraphNodeDto[], edges: FaceGraphEdgeDto[]) =>
  createLayout(nodes, edges).getPositions();

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
  const query = normalizeSearchString(name.trim());
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

export const SLIDER_STEPS = 100;

/**
 * Most people have few photos and a few people have very many, so the photo slider is
 * logarithmic: position 0 is one photo and the last position is the largest person.
 */
export const sliderToPhotos = (position: number, limit: number) =>
  Math.round(Math.max(limit, 1) ** (Math.min(Math.max(position, 0), SLIDER_STEPS) / SLIDER_STEPS));

export const photosToSlider = (photos: number, limit: number) =>
  limit <= 1 ? 0 : Math.round((Math.log(Math.min(Math.max(photos, 1), limit)) / Math.log(limit)) * SLIDER_STEPS);
