import { beforeEach, describe, expect, it } from 'vitest';
import { FaceGraphNodeKind } from 'src/dtos/face-graph.dto.js';
import { FaceGraphService } from 'src/services/face-graph.service.js';
import { authStub } from 'test/fixtures/auth.stub.js';
import { newUuid } from 'test/small.factory.js';
import { ServiceMocks, newTestService } from 'test/utils.js';

const centroid = (...values: number[]) => JSON.stringify(values);

const person = (values: { name?: string; centroid: string; assetCount?: number; faceCount?: number }) => ({
  personGroupId: newUuid(),
  name: '',
  isHidden: false,
  isFavorite: false,
  updatedAt: new Date('2026-01-01T00:00:00.000Z'),
  assetCount: 1,
  faceCount: 1,
  ...values,
});

describe(FaceGraphService.name, () => {
  let sut: FaceGraphService;
  let mocks: ServiceMocks;

  beforeEach(() => {
    ({ sut, mocks } = newTestService(FaceGraphService));
  });

  it('should work', () => {
    expect(sut).toBeDefined();
  });

  describe('getGraph', () => {
    it('should return an empty graph when there are no people', async () => {
      mocks.person.getCentroids.mockResolvedValue([]);

      await expect(sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5 })).resolves.toEqual({
        nodes: [],
        edges: [],
      });
      expect(mocks.person.getCentroids).toHaveBeenCalledWith(authStub.admin.user.id, {
        minFaces: 1,
        withHidden: false,
      });
    });

    it('should map people to nodes', async () => {
      const alice = person({ name: 'Alice', centroid: centroid(1, 0, 0), assetCount: 12, faceCount: 15 });
      mocks.person.getCentroids.mockResolvedValue([alice]);

      await expect(sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5 })).resolves.toEqual({
        nodes: [
          {
            id: alice.personGroupId,
            kind: FaceGraphNodeKind.Person,
            name: 'Alice',
            isHidden: false,
            isFavorite: false,
            updatedAt: '2026-01-01T00:00:00.000Z',
            assetCount: 12,
            faceCount: 15,
            x: 0,
            y: 0,
          },
        ],
        edges: [],
      });
    });

    it('should link similar people only', async () => {
      const alice = person({ centroid: centroid(1, 0, 0) });
      const alsoAlice = person({ centroid: centroid(2, 0.2, 0) });
      const bob = person({ centroid: centroid(0, 0, 3) });
      mocks.person.getCentroids.mockResolvedValue([alice, alsoAlice, bob]);

      const { nodes, edges } = await sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5, maxDistance: 0.5 });

      expect(nodes).toHaveLength(3);
      expect(edges).toEqual([
        { source: alice.personGroupId, target: alsoAlice.personGroupId, distance: expect.closeTo(0.005, 3) },
      ]);
    });

    it('should pass the filters to the repository', async () => {
      mocks.person.getCentroids.mockResolvedValue([]);

      await sut.getGraph(authStub.admin, { minFaces: 3, withHidden: true, neighbors: 5 });

      expect(mocks.person.getCentroids).toHaveBeenCalledWith(authStub.admin.user.id, {
        minFaces: 3,
        withHidden: true,
      });
    });
  });
});
