import {
  clusterFaces,
  cosineDistance,
  meanVector,
  nearestNeighbors,
  normalize,
  parseVector,
  projectTo2d,
} from 'src/utils/face-graph';
import { describe, expect, it } from 'vitest';

const SIZE = 16;

/** deterministic pseudo random numbers in [-1, 1] */
const random = (seed: number) => {
  let state = seed;
  return () => {
    state = (state * 1_664_525 + 1_013_904_223) % 4_294_967_296;
    return (state / 4_294_967_296) * 2 - 1;
  };
};

/** a unit vector pointing mostly along `axis`, with a bit of noise */
const face = (axis: number, next: () => number, noise = 0.05) =>
  normalize(Float32Array.from({ length: SIZE }, (_, i) => (i === axis ? 1 : 0) + next() * noise));

const faces = (axis: number, count: number, seed: number) => {
  const next = random(seed);
  return Array.from({ length: count }, () => face(axis, next));
};

describe('face graph utils', () => {
  describe('parseVector', () => {
    it('should parse a pgvector string', () => {
      expect(parseVector('[1,0.5,-2]').values().toArray()).toEqual([1, 0.5, -2]);
    });
  });

  describe('meanVector', () => {
    it('should return a unit vector', () => {
      const mean = meanVector([Float32Array.from([2, 0]), Float32Array.from([0, 2])]);
      expect(mean[0]).toBeCloseTo(Math.SQRT1_2);
      expect(mean[1]).toBeCloseTo(Math.SQRT1_2);
    });

    it('should only use the given members', () => {
      const mean = meanVector([Float32Array.from([1, 0]), Float32Array.from([0, 1])], [1]);
      expect(mean.values().toArray()).toEqual([0, 1]);
    });
  });

  describe('nearestNeighbors', () => {
    it('should only link vectors within the maximum distance', () => {
      const vectors = [...faces(0, 3, 1), ...faces(1, 3, 2)];
      const edges = nearestNeighbors(vectors, { k: 5, maxDistance: 0.5 });

      expect(edges).toHaveLength(6);
      for (const { source, target, distance } of edges) {
        expect(source).toBeLessThan(target);
        expect(source < 3).toBe(target < 3);
        expect(distance).toBeCloseTo(cosineDistance(vectors[source], vectors[target]));
      }
    });

    it('should keep at most k neighbors per vector', () => {
      const vectors = faces(0, 10, 3);
      const edges = nearestNeighbors(vectors, { k: 1, maxDistance: 2 });

      // every vector picks one neighbor, mutual picks are merged
      expect(edges.length).toBeGreaterThanOrEqual(5);
      expect(edges.length).toBeLessThanOrEqual(10);
    });

    it('should handle an empty list', () => {
      expect(nearestNeighbors([], { k: 5, maxDistance: 1 })).toEqual([]);
    });
  });

  describe('projectTo2d', () => {
    it('should be deterministic', () => {
      const vectors = [...faces(0, 5, 1), ...faces(1, 5, 2), ...faces(2, 5, 3)];
      expect(projectTo2d(vectors)).toEqual(projectTo2d(vectors));
    });

    it('should keep similar vectors close together', () => {
      const vectors = [...faces(0, 5, 1), ...faces(1, 5, 2), ...faces(2, 5, 3)];
      const points = projectTo2d(vectors);
      const distance = (a: number, b: number) => Math.hypot(points[a].x - points[b].x, points[a].y - points[b].y);

      expect(distance(0, 1)).toBeLessThan(distance(0, 5));
      expect(distance(5, 6)).toBeLessThan(distance(5, 10));
      for (const { x, y } of points) {
        expect(Math.abs(x)).toBeLessThanOrEqual(1);
        expect(Math.abs(y)).toBeLessThanOrEqual(1);
      }
    });

    it('should handle empty and single inputs', () => {
      expect(projectTo2d([])).toEqual([]);
      expect(projectTo2d(faces(0, 1, 1))).toEqual([{ x: 0, y: 0 }]);
    });
  });

  describe('clusterFaces', () => {
    it('should separate two different identities', () => {
      const vectors = [...faces(0, 20, 1), ...faces(1, 5, 2)];
      const clusters = clusterFaces(vectors, 0.4);

      expect(clusters).toHaveLength(2);
      expect(clusters[0].members.toSorted((a, b) => a - b)).toEqual(Array.from({ length: 20 }, (_, i) => i));
      expect(clusters[1].members.toSorted((a, b) => a - b)).toEqual([20, 21, 22, 23, 24]);
    });

    it('should keep a single identity together', () => {
      expect(clusterFaces(faces(0, 20, 1), 0.4)).toHaveLength(1);
    });

    it('should not create more groups with a higher threshold', () => {
      const next = random(7);
      const vectors = Array.from({ length: 60 }, (_, i) => face(i % 4, next, 0.6));
      const counts = [0.1, 0.3, 0.5, 0.8, 1.5].map((threshold) => clusterFaces(vectors, threshold).length);

      expect(counts).toEqual(counts.toSorted((a, b) => b - a));
      expect(counts.at(-1)).toBe(1);
    });

    it('should assign every vector to exactly one group', () => {
      const vectors = [...faces(0, 7, 1), ...faces(1, 7, 2), ...faces(2, 7, 3)];
      const members = clusterFaces(vectors, 0.4).flatMap((cluster) => cluster.members);

      expect(members.toSorted((a, b) => a - b)).toEqual(Array.from({ length: 21 }, (_, i) => i));
    });

    it('should handle an empty list', () => {
      expect(clusterFaces([], 0.4)).toEqual([]);
    });
  });
});
