import { BadRequestException, Injectable } from '@nestjs/common';
import { AuthDto } from 'src/dtos/auth.dto';
import {
  FaceGraphDto,
  FaceGraphNodeKind,
  FaceGraphResponseDto,
  FaceGroupsDto,
  FaceGroupsResponseDto,
} from 'src/dtos/face-graph.dto';
import { Permission } from 'src/enum';
import { BaseService } from 'src/services/base.service';
import { asDateTimeString } from 'src/utils/date';
import {
  type FaceCluster,
  clusterFaces,
  cosineDistance,
  nearestNeighbors,
  normalize,
  parseVector,
  projectTo2d,
} from 'src/utils/face-graph';

type PersonCentroid = {
  personGroupId: string;
  name: string;
  isHidden: boolean;
  isFavorite: boolean;
  updatedAt: Date;
  assetCount: number;
  faceCount: number;
  centroid: Float32Array;
};

type Face = {
  id: string;
  assetId: string;
  imageWidth: number;
  imageHeight: number;
  boundingBoxX1: number;
  boundingBoxY1: number;
  boundingBoxX2: number;
  boundingBoxY2: number;
  fileCreatedAt: Date;
};

type Faces = { faces: Face[]; vectors: Float32Array[]; truncated: boolean };

// grouping the faces of a person compares against everyone else, which is expensive to recompute for every request
const CENTROID_CACHE_MS = 60_000;
const MAX_FACES = 5000;
// unassigned faces rarely look alike, which makes grouping them slow
const MAX_UNASSIGNED_FACES = 2000;
const UNASSIGNED_THRESHOLD = 0.4;
const UNASSIGNED_MIN_FACES = 2;

const mapFace = (face: Face, distance: number) => ({
  id: face.id,
  assetId: face.assetId,
  fileCreatedAt: asDateTimeString(face.fileCreatedAt),
  distance,
  imageWidth: face.imageWidth,
  imageHeight: face.imageHeight,
  boundingBoxX1: face.boundingBoxX1,
  boundingBoxY1: face.boundingBoxY1,
  boundingBoxX2: face.boundingBoxX2,
  boundingBoxY2: face.boundingBoxY2,
});

/** the face that looks most like the rest of the cluster */
const getLeader = (cluster: FaceCluster, vectors: Float32Array[]) => {
  let leader = cluster.members[0];
  let leaderDistance = Infinity;
  for (const index of cluster.members) {
    const distance = cosineDistance(vectors[index], cluster.centroid);
    if (!(distance < leaderDistance)) {
      continue;
    }

    leader = index;
    leaderDistance = distance;
  }
  return leader;
};

const getClosestPerson = (centroid: Float32Array, people: PersonCentroid[]) => {
  let closest: { id: string; name: string; distance: number } | null = null;
  for (const person of people) {
    const distance = cosineDistance(centroid, person.centroid);
    if (!closest || distance < closest.distance) {
      closest = { id: person.personGroupId, name: person.name, distance };
    }
  }
  return closest;
};

@Injectable()
export class FaceGraphService extends BaseService {
  private centroidCache = new Map<string, { expiresAt: number; centroids: Promise<PersonCentroid[]> }>();

  async getGraph(auth: AuthDto, dto: FaceGraphDto): Promise<FaceGraphResponseDto> {
    const { machineLearning } = await this.getConfig({ withCache: true });
    const { minFaces, withHidden = false, withUnassigned = false, neighbors, maxDistance } = dto;

    const centroids = await this.getCentroids(auth.user.id, { refresh: true });
    const people = centroids.filter((person) => person.faceCount >= minFaces && (withHidden || !person.isHidden));
    const nodes = people.map((person) => ({
      id: person.personGroupId,
      kind: FaceGraphNodeKind.Person,
      face: null as ReturnType<typeof mapFace> | null,
      name: person.name,
      isHidden: person.isHidden,
      isFavorite: person.isFavorite,
      updatedAt: asDateTimeString(person.updatedAt),
      assetCount: person.assetCount,
      faceCount: person.faceCount,
    }));
    const vectors = people.map(({ centroid }) => centroid);

    if (withUnassigned) {
      const unassigned = await this.getUnassignedClusters(auth.user.id);
      for (const cluster of unassigned.clusters) {
        const leader = unassigned.faces[getLeader(cluster, unassigned.vectors)];
        nodes.push({
          id: leader.id,
          kind: FaceGraphNodeKind.Unassigned,
          face: mapFace(leader, 0),
          name: '',
          isHidden: false,
          isFavorite: false,
          updatedAt: asDateTimeString(leader.fileCreatedAt),
          assetCount: new Set(cluster.members.map((index) => unassigned.faces[index].assetId)).size,
          faceCount: cluster.members.length,
        });
        vectors.push(cluster.centroid);
      }
    }

    const positions = projectTo2d(vectors);
    const edges = nearestNeighbors(vectors, {
      k: neighbors,
      maxDistance: maxDistance ?? machineLearning.facialRecognition.maxDistance,
    });

    return {
      nodes: nodes.map((node, index) => ({ ...node, x: positions[index].x, y: positions[index].y })),
      edges: edges.map(({ source, target, distance }) => ({
        source: nodes[source].id,
        target: nodes[target].id,
        distance,
      })),
    };
  }

  async getGroups(auth: AuthDto, personGroupId: string, { threshold }: FaceGroupsDto): Promise<FaceGroupsResponseDto> {
    await this.requireAccess({ auth, permission: Permission.PersonRead, ids: [personGroupId] });

    const { faces, vectors, truncated } = await this.getFaces(auth.user.id, personGroupId, MAX_FACES);
    if (faces.length === 0) {
      return { threshold, truncated, groups: [] };
    }

    const centroids = await this.getCentroids(auth.user.id);
    const others = centroids.filter((person) => person.personGroupId !== personGroupId);
    const clusters = clusterFaces(vectors, threshold);
    const [main] = clusters;

    return {
      threshold,
      truncated,
      groups: clusters.map((cluster) =>
        this.mapGroup(cluster, main, { faces, vectors, people: cluster === main ? [] : others }),
      ),
    };
  }

  async getUnassignedGroups(
    auth: AuthDto,
    faceId: string,
    { threshold }: FaceGroupsDto,
  ): Promise<FaceGroupsResponseDto> {
    const unassigned = await this.getUnassignedClusters(auth.user.id);
    const index = unassigned.faces.findIndex(({ id }) => id === faceId);
    const cluster = unassigned.clusters.find(({ members }) => members.includes(index));
    if (!cluster) {
      throw new BadRequestException('Unassigned faces not found');
    }

    const faces = cluster.members.map((member) => unassigned.faces[member]);
    const vectors = cluster.members.map((member) => unassigned.vectors[member]);
    const people = await this.getCentroids(auth.user.id);
    const clusters = clusterFaces(vectors, threshold);
    const [main] = clusters;

    return {
      threshold,
      truncated: false,
      groups: clusters.map((cluster) => this.mapGroup(cluster, main, { faces, vectors, people })),
    };
  }

  private mapGroup(
    cluster: FaceCluster,
    main: FaceCluster,
    { faces, vectors, people }: { faces: Face[]; vectors: Float32Array[]; people: PersonCentroid[] },
  ) {
    const members = cluster.members
      .map((index) => ({
        face: faces[index],
        vector: vectors[index],
        distance: cosineDistance(vectors[index], cluster.centroid),
      }))
      .toSorted((a, b) => a.distance - b.distance);

    return {
      id: members[0].face.id,
      assetCount: new Set(members.map(({ face }) => face.assetId)).size,
      distanceToMain: cluster === main ? 0 : cosineDistance(cluster.centroid, main.centroid),
      closestPerson: getClosestPerson(cluster.centroid, people),
      faces: members.map(({ face, vector }) => mapFace(face, cosineDistance(vector, main.centroid))),
    };
  }

  private async getFaces(userId: string, personGroupId: string | null, limit: number): Promise<Faces> {
    const rows = await this.personRepository.getFaceEmbeddings({ userId, personGroupId, limit: limit + 1 });
    const faces = rows.slice(0, limit);
    return {
      faces,
      vectors: faces.map(({ embedding }) => normalize(parseVector(embedding))),
      truncated: rows.length > limit,
    };
  }

  private async getUnassignedClusters(userId: string) {
    const { faces, vectors } = await this.getFaces(userId, null, MAX_UNASSIGNED_FACES);
    const clusters = clusterFaces(vectors, UNASSIGNED_THRESHOLD).filter(
      ({ members }) => members.length >= UNASSIGNED_MIN_FACES,
    );
    return { faces, vectors, clusters };
  }

  private getCentroids(userId: string, { refresh = false }: { refresh?: boolean } = {}) {
    const cached = this.centroidCache.get(userId);
    if (cached && !refresh && cached.expiresAt > Date.now()) {
      return cached.centroids;
    }

    const centroids = this.loadCentroids(userId);
    this.centroidCache.set(userId, { expiresAt: Date.now() + CENTROID_CACHE_MS, centroids });
    centroids.catch(() => this.centroidCache.delete(userId));
    return centroids;
  }

  private async loadCentroids(userId: string): Promise<PersonCentroid[]> {
    const people = await this.personRepository.getCentroids(userId);
    return people.map((person) => ({
      ...person,
      assetCount: Number(person.assetCount),
      faceCount: Number(person.faceCount),
      centroid: normalize(parseVector(person.centroid)),
    }));
  }
}
