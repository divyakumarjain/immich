import { Injectable } from '@nestjs/common';
import { AuthDto } from 'src/dtos/auth.dto.js';
import { FaceGraphDto, FaceGraphNodeKind, FaceGraphResponseDto } from 'src/dtos/face-graph.dto.js';
import { BaseService } from 'src/services/base.service.js';
import { asDateTimeString } from 'src/utils/date.js';
import { nearestNeighbors, normalize, parseVector, projectTo2d } from 'src/utils/face-graph.js';

@Injectable()
export class FaceGraphService extends BaseService {
  async getGraph(auth: AuthDto, dto: FaceGraphDto): Promise<FaceGraphResponseDto> {
    const { machineLearning } = await this.getConfig({ withCache: true });
    const { minFaces, withHidden = false, neighbors, maxDistance } = dto;

    const people = await this.personRepository.getCentroids(auth.user.id, { minFaces, withHidden });
    const centroids = people.map(({ centroid }) => normalize(parseVector(centroid)));
    const positions = projectTo2d(centroids);
    const edges = nearestNeighbors(centroids, {
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
        assetCount: Number(person.assetCount),
        faceCount: Number(person.faceCount),
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
}
