export type FaceGraphEdge = { source: number; target: number; distance: number };
export type FaceGraphPoint = { x: number; y: number };
export type FaceCluster = { members: number[]; centroid: Float32Array };

export interface NearestNeighborOptions {
  /** maximum number of neighbors per vector */
  k: number;
  /** maximum cosine distance for two vectors to be considered neighbors */
  maxDistance: number;
}

const POWER_ITERATIONS = 30;

/** parses the text representation of a pgvector value, e.g. `[0.1,0.2]` */
export const parseVector = (value: string): Float32Array => Float32Array.from(JSON.parse(value) as number[]);

const dot = (a: Float32Array, b: Float32Array): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
};

/** scales the vector to unit length in place */
export const normalize = (vector: Float32Array): Float32Array => {
  const norm = Math.sqrt(dot(vector, vector));
  if (norm > 0) {
    for (let i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }
  return vector;
};

/** cosine distance between two unit vectors */
export const cosineDistance = (a: Float32Array, b: Float32Array): number => 1 - dot(a, b);

/** unit-length mean of the given vectors */
export const meanVector = (vectors: Float32Array[], members?: number[]): Float32Array => {
  const indexes = members ?? vectors.keys().toArray();
  const mean = new Float32Array(vectors[indexes[0]]?.length ?? 0);
  for (const index of indexes) {
    const vector = vectors[index];
    for (let i = 0; i < mean.length; i++) {
      mean[i] += vector[i];
    }
  }
  return normalize(mean);
};

/**
 * Finds, for every unit vector, its closest vectors within `maxDistance`.
 * Each pair is returned once, with `source < target`.
 */
export const nearestNeighbors = (vectors: Float32Array[], { k, maxDistance }: NearestNeighborOptions) => {
  const neighbors: { index: number; distance: number }[][] = vectors.map(() => []);
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) {
      const distance = cosineDistance(vectors[i], vectors[j]);
      if (!(distance <= maxDistance)) {
        continue;
      }

      neighbors[i].push({ index: j, distance });
      neighbors[j].push({ index: i, distance });
    }
  }

  const edges = new Map<string, FaceGraphEdge>();
  for (const [i, candidates] of neighbors.entries()) {
    candidates.sort((a, b) => a.distance - b.distance);
    for (const { index, distance } of candidates.slice(0, k)) {
      const source = Math.min(i, index);
      const target = Math.max(i, index);
      edges.set(`${source}-${target}`, { source, target, distance });
    }
  }

  return edges.values().toArray();
};

/**
 * Projects the vectors onto their first two principal components.
 * The result is deterministic and scaled to fit within [-1, 1].
 */
export const projectTo2d = (vectors: Float32Array[]): FaceGraphPoint[] => {
  if (vectors.length === 0) {
    return [];
  }

  const size = vectors[0].length;
  const mean = new Float32Array(size);
  for (const vector of vectors) {
    for (let i = 0; i < size; i++) {
      mean[i] += vector[i] / vectors.length;
    }
  }
  const centered = vectors.map((vector) => vector.map((value, i) => value - mean[i]));

  const components: Float32Array[] = [];
  for (let c = 0; c < 2; c++) {
    let component = normalize(Float32Array.from({ length: size }, (_, i) => Math.sin((i + 1) * (c + 1))));
    for (let iteration = 0; iteration < POWER_ITERATIONS; iteration++) {
      const next = new Float32Array(size);
      for (const vector of centered) {
        const projection = dot(vector, component);
        for (let i = 0; i < size; i++) {
          next[i] += projection * vector[i];
        }
      }
      for (const previous of components) {
        const overlap = dot(next, previous);
        for (let i = 0; i < size; i++) {
          next[i] -= overlap * previous[i];
        }
      }
      component = normalize(next);
    }
    components.push(component);
  }

  const points = centered.map((vector) => ({ x: dot(vector, components[0]), y: dot(vector, components[1]) }));
  const scale = Math.max(...points.map(({ x, y }) => Math.max(Math.abs(x), Math.abs(y))));
  return scale > 0 ? points.map(({ x, y }) => ({ x: x / scale, y: y / scale })) : points;
};

const closestCluster = (vector: Float32Array, clusters: FaceCluster[]) => {
  let closest = -1;
  let closestDistance = Infinity;
  for (const [index, cluster] of clusters.entries()) {
    const distance = cosineDistance(vector, cluster.centroid);
    if (!(distance < closestDistance)) {
      continue;
    }

    closest = index;
    closestDistance = distance;
  }
  return { index: closest, distance: closestDistance };
};

/**
 * Groups unit vectors so that each one is within `threshold` of its group centroid.
 * Vectors closest to the overall centroid are placed first, so the first groups are the most representative.
 * Groups are returned largest first.
 */
export const clusterFaces = (vectors: Float32Array[], threshold: number): FaceCluster[] => {
  if (vectors.length === 0) {
    return [];
  }

  const center = meanVector(vectors);
  const order = vectors
    .map((vector, index) => ({ index, distance: cosineDistance(vector, center) }))
    .sort((a, b) => a.distance - b.distance)
    .map(({ index }) => index);

  let clusters: FaceCluster[] = [];
  for (const index of order) {
    const closest = closestCluster(vectors[index], clusters);
    if (closest.index !== -1 && closest.distance <= threshold) {
      const cluster = clusters[closest.index];
      cluster.members.push(index);
      cluster.centroid = meanVector(vectors, cluster.members);
    } else {
      clusters.push({ members: [index], centroid: Float32Array.from(vectors[index]) });
    }
  }

  // settle: the centroids moved while the groups were growing
  const members: number[][] = clusters.map(() => []);
  for (const index of order) {
    members[closestCluster(vectors[index], clusters).index].push(index);
  }
  clusters = members
    .filter((group) => group.length > 0)
    .map((group) => ({ members: group, centroid: meanVector(vectors, group) }));

  return clusters.sort((a, b) => b.members.length - a.members.length);
};
