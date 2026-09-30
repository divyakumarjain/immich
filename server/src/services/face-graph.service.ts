import { Injectable } from '@nestjs/common';
import { AuthDto } from 'src/dtos/auth.dto.js';
import {
  FaceGraphDto,
  FaceGraphNodeKind,
  FaceGraphResponseDto,
  FaceGroupsDto,
  FaceGroupsResponseDto,
} from 'src/dtos/face-graph.dto.js';
import { Permission } from 'src/enum.js';
import { BaseService } from 'src/services/base.service.js';
import { asDateTimeString } from 'src/utils/date.js';
import {
  clusterFaces,
  cosineDistance,
  nearestNeighbors,
  normalize,
  parseVector,
  projectTo2d,
} from 'src/utils/face-graph.js';

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

// grouping the faces of a person compares against everyone else, which is expensive to recompute for every request
const CENTROID_CACHE_MS = 60_000;
const MAX_FACES = 5000;

@Injectable()
export class FaceGraphService extends BaseService {
  private centroidCache = new Map<string, { expiresAt: number; centroids: Promise<PersonCentroid[]> }>();

  async getGraph(auth: AuthDto, dto: FaceGraphDto): Promise<FaceGraphResponseDto> {
    const { machineLearning } = await this.getConfig({ withCache: true });
    const { minFaces, withHidden = false, neighbors, maxDistance } = dto;

    const centroids = await this.getCentroids(auth.user.id, { refresh: true });
    const people = centroids.filter((person) => person.faceCount >= minFaces && (withHidden || !person.isHidden));
    const vectors = people.map(({ centroid }) => centroid);
    const positions = projectTo2d(vectors);
    const edges = nearestNeighbors(vectors, {
      k: neighbors,
      maxDistance: maxDistance ?? machineLearning.facialRecognition.maxDistance,
    });

    return {
      nodes: people.map((person, index) => ({
        id: person.personGroupId,
        kind: FaceGraphNodeKind.Person,
        name: person.name,
        isHidden: person.isHidden,
        isFavorite: person.isFavorite,
        updatedAt: asDateTimeString(person.updatedAt),
        assetCount: person.assetCount,
        faceCount: person.faceCount,
        x: positions[index].x,
        y: positions[index].y,
      })),
      edges: edges.map(({ source, target, distance }) => ({
        source: people[source].personGroupId,
        target: people[target].personGroupId,
        distance,
      })),
    };
  }

  async getGroups(auth: AuthDto, personGroupId: string, { threshold }: FaceGroupsDto): Promise<FaceGroupsResponseDto> {
    await this.requirePersonAccess({
      auth,
      permission: Permission.PersonRead,
      ids: [{ personGroupId, ownerId: auth.user.id }],
    });

    const rows = await this.personRepository.getFaceEmbeddings({
      userId: auth.user.id,
      personGroupId,
      limit: MAX_FACES + 1,
    });
    const faces = rows.slice(0, MAX_FACES);
    const vectors = faces.map(({ embedding }) => normalize(parseVector(embedding)));
    const clusters = clusterFaces(vectors, threshold);
    if (clusters.length === 0) {
      return { threshold, truncated: false, groups: [] };
    }

    const [main] = clusters;
    const centroids = await this.getCentroids(auth.user.id);
    const others = centroids.filter((person) => person.personGroupId !== personGroupId);

    return {
      threshold,
      truncated: rows.length > MAX_FACES,
      groups: clusters.map((cluster) => {
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
          closestPerson: cluster === main ? null : this.getClosestPerson(cluster.centroid, others),
          faces: members.map(({ face, vector }) => ({
            id: face.id,
            assetId: face.assetId,
            fileCreatedAt: asDateTimeString(face.fileCreatedAt),
            distance: cosineDistance(vector, main.centroid),
            imageWidth: face.imageWidth,
            imageHeight: face.imageHeight,
            boundingBoxX1: face.boundingBoxX1,
            boundingBoxY1: face.boundingBoxY1,
            boundingBoxX2: face.boundingBoxX2,
            boundingBoxY2: face.boundingBoxY2,
          })),
        };
      }),
    };
  }

  private getClosestPerson(centroid: Float32Array, people: PersonCentroid[]) {
    let closest: { id: string; name: string; distance: number } | null = null;
    for (const person of people) {
      const distance = cosineDistance(centroid, person.centroid);
      if (!closest || distance < closest.distance) {
        closest = { id: person.personGroupId, name: person.name, distance };
      }
    }
    return closest;
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
