import { FaceGraphNodeKind, type FaceGraphNodeDto } from '@immich/sdk';
import {
  computeLayout,
  getNeighbors,
  getWorkQueue,
  matchesFilters,
  nodeRadius,
  searchNodes,
  type FaceGraphFilters,
} from '$lib/utils/face-graph';

const node = (id: string, values: Partial<FaceGraphNodeDto> = {}): FaceGraphNodeDto => ({
  id,
  kind: FaceGraphNodeKind.Person,
  name: '',
  isHidden: false,
  isFavorite: false,
  updatedAt: '2026-01-01T00:00:00.000Z',
  assetCount: 10,
  faceCount: 10,
  x: 0,
  y: 0,
  ...values,
});

const filters: FaceGraphFilters = { unnamedOnly: false, showHidden: false, minPhotos: 1 };

describe('face graph utils', () => {
  describe('nodeRadius', () => {
    it('should grow with the square root of the number of photos', () => {
      expect(nodeRadius(100)).toBe(20);
      expect(nodeRadius(400)).toBe(40);
    });

    it('should stay within limits', () => {
      expect(nodeRadius(0)).toBe(6);
      expect(nodeRadius(100_000)).toBe(60);
    });
  });

  describe('matchesFilters', () => {
    it('should hide hidden people by default', () => {
      expect(matchesFilters(node('a', { isHidden: true }), filters)).toBe(false);
      expect(matchesFilters(node('a', { isHidden: true }), { ...filters, showHidden: true })).toBe(true);
    });

    it('should only keep unnamed people when asked', () => {
      expect(matchesFilters(node('a', { name: 'Alice' }), { ...filters, unnamedOnly: true })).toBe(false);
      expect(matchesFilters(node('a'), { ...filters, unnamedOnly: true })).toBe(true);
    });

    it('should apply the minimum number of photos', () => {
      expect(matchesFilters(node('a', { assetCount: 4 }), { ...filters, minPhotos: 5 })).toBe(false);
      expect(matchesFilters(node('a', { assetCount: 5 }), { ...filters, minPhotos: 5 })).toBe(true);
    });
  });

  describe('searchNodes', () => {
    const nodes = [
      node('a', { name: 'Zoé', assetCount: 1 }),
      node('b', { name: 'Zoe Smith', assetCount: 5 }),
      node('c'),
    ];

    it('should ignore case and accents', () => {
      expect(searchNodes(nodes, 'zoe').map(({ id }) => id)).toEqual(['b', 'a']);
    });

    it('should return nothing for an empty search', () => {
      expect(searchNodes(nodes, ' ')).toEqual([]);
    });
  });

  describe('getWorkQueue', () => {
    it('should list unnamed people with the most photos first', () => {
      const nodes = [
        node('a', { assetCount: 3 }),
        node('b', { assetCount: 30, name: 'Bob' }),
        node('c', { assetCount: 20 }),
      ];
      expect(getWorkQueue(nodes).map(({ id }) => id)).toEqual(['c', 'a']);
    });
  });

  describe('getNeighbors', () => {
    it('should return linked nodes, most similar first', () => {
      const edges = [
        { source: 'a', target: 'b', distance: 0.4 },
        { source: 'c', target: 'a', distance: 0.1 },
        { source: 'b', target: 'c', distance: 0.2 },
      ];
      expect(getNeighbors('a', edges)).toEqual([
        { id: 'c', distance: 0.1 },
        { id: 'b', distance: 0.4 },
      ]);
    });
  });

  describe('computeLayout', () => {
    const distance = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(a.x - b.x, a.y - b.y);

    it('should not let nodes overlap', () => {
      const nodes = Array.from({ length: 20 }, (_, i) => node(String(i), { assetCount: 100 }));
      const positions = computeLayout(nodes, []);

      for (const a of nodes) {
        for (const b of nodes) {
          if (a.id !== b.id) {
            expect(distance(positions.get(a.id)!, positions.get(b.id)!)).toBeGreaterThan(39);
          }
        }
      }
    });

    it('should keep linked nodes closer than unrelated ones', () => {
      const nodes = [node('a', { x: -1, y: 0 }), node('b', { x: 1, y: 0 }), node('c', { x: 0, y: 1 })];
      const positions = computeLayout(nodes, [{ source: 'a', target: 'b', distance: 0.1 }]);

      expect(distance(positions.get('a')!, positions.get('b')!)).toBeLessThan(
        distance(positions.get('a')!, positions.get('c')!),
      );
    });

    it('should ignore links to unknown nodes', () => {
      const positions = computeLayout([node('a')], [{ source: 'a', target: 'missing', distance: 0.1 }]);
      expect(positions.size).toBe(1);
    });
  });
});
