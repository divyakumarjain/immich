import type { FaceGraphEdgeDto, FaceGraphNodeDto } from '@immich/sdk';
import { forceCollide, forceLink, forceSimulation, forceX, forceY } from 'd3-force';
import { normalizeSearchString } from '$lib/utils/string-utils';

export type FaceGraphPosition = { x: number; y: number };

export type FaceGraphFilters = {
  unnamedOnly: boolean;
  showHidden: boolean;
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

export const matchesFilters = (node: FaceGraphNodeDto, filters: FaceGraphFilters) => {
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
