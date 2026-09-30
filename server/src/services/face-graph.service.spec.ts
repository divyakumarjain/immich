import { BadRequestException } from '@nestjs/common';
import { FaceGraphNodeKind } from 'src/dtos/face-graph.dto';
import { FaceGraphService } from 'src/services/face-graph.service';
import { authStub } from 'test/fixtures/auth.stub';
import { newUuid } from 'test/small.factory';
import { ServiceMocks, newTestService } from 'test/utils';
import { beforeEach, describe, expect, it } from 'vitest';

const centroid = (...values: number[]) => JSON.stringify(values);

const person = (values: {
  name?: string;
  centroid: string;
  assetCount?: number;
  faceCount?: number;
  isHidden?: boolean;
}) => ({
  personGroupId: newUuid(),
  name: '',
  isHidden: false,
  isFavorite: false,
  updatedAt: new Date('2026-01-01T00:00:00.000Z'),
  assetCount: 1,
  faceCount: 1,
  ...values,
});

const face = (embedding: string, assetId = newUuid()) => ({
  id: newUuid(),
  assetId,
  imageWidth: 1000,
  imageHeight: 800,
  boundingBoxX1: 10,
  boundingBoxY1: 20,
  boundingBoxX2: 110,
  boundingBoxY2: 140,
  fileCreatedAt: new Date('2026-01-01T00:00:00.000Z'),
  embedding,
});

describe(FaceGraphService.name, () => {
  let sut: FaceGraphService;
  let mocks: ServiceMocks;

  beforeEach(() => {
    ({ sut, mocks } = newTestService(FaceGraphService));
  });

  const allowAccess = (personGroupId: string) =>
    mocks.access.person.checkOwnerAccess.mockResolvedValue(new Set([personGroupId]));

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
      expect(mocks.person.getCentroids).toHaveBeenCalledWith(authStub.admin.user.id);
    });

    it('should map people to nodes', async () => {
      const alice = person({ name: 'Alice', centroid: centroid(1, 0, 0), assetCount: 12, faceCount: 15 });
      mocks.person.getCentroids.mockResolvedValue([alice]);

      await expect(sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5 })).resolves.toEqual({
        nodes: [
          {
            id: alice.personGroupId,
            kind: FaceGraphNodeKind.Person,
            face: null,
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

    it('should leave out hidden people and people with too few faces', async () => {
      const alice = person({ centroid: centroid(1, 0, 0), faceCount: 5 });
      const hidden = person({ centroid: centroid(0, 1, 0), faceCount: 5, isHidden: true });
      const small = person({ centroid: centroid(0, 0, 1), faceCount: 1 });
      mocks.person.getCentroids.mockResolvedValue([alice, hidden, small]);

      const { nodes } = await sut.getGraph(authStub.admin, { minFaces: 3, neighbors: 5 });
      expect(nodes.map(({ id }) => id)).toEqual([alice.personGroupId]);

      const withHidden = await sut.getGraph(authStub.admin, { minFaces: 3, withHidden: true, neighbors: 5 });
      expect(withHidden.nodes.map(({ id }) => id)).toEqual([alice.personGroupId, hidden.personGroupId]);
    });

    it('should always load the latest people', async () => {
      mocks.person.getCentroids.mockResolvedValue([]);

      await sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5 });
      await sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5 });

      expect(mocks.person.getCentroids).toHaveBeenCalledTimes(2);
    });
  });

  describe('getGraph with unassigned faces', () => {
    it('should not load unassigned faces by default', async () => {
      mocks.person.getCentroids.mockResolvedValue([]);

      await sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5 });

      expect(mocks.person.getFaceEmbeddings).not.toHaveBeenCalled();
    });

    it('should add groups of similar unassigned faces', async () => {
      const alice = person({ centroid: centroid(1, 0, 0) });
      const stranger = [face(centroid(0, 1, 0)), face(centroid(0, 1, 0.1)), face(centroid(0, 1, -0.1))];
      const single = face(centroid(0, 0, 1));
      mocks.person.getCentroids.mockResolvedValue([alice]);
      mocks.person.getFaceEmbeddings.mockResolvedValue([single, ...stranger]);

      const { nodes } = await sut.getGraph(authStub.admin, { minFaces: 1, neighbors: 5, withUnassigned: true });

      expect(mocks.person.getFaceEmbeddings).toHaveBeenCalledWith({
        userId: authStub.admin.user.id,
        personGroupId: null,
        limit: 2001,
      });
      expect(nodes).toHaveLength(2);
      expect(nodes[1]).toEqual(
        expect.objectContaining({
          id: stranger[0].id,
          kind: FaceGraphNodeKind.Unassigned,
          name: '',
          assetCount: 3,
          faceCount: 3,
          face: expect.objectContaining({ id: stranger[0].id, assetId: stranger[0].assetId, boundingBoxX1: 10 }),
        }),
      );
    });
  });

  describe('getUnassignedGroups', () => {
    it('should fail for a face that is not part of a group', async () => {
      const single = face(centroid(0, 0, 1));
      mocks.person.getFaceEmbeddings.mockResolvedValue([single]);

      await expect(sut.getUnassignedGroups(authStub.admin, single.id, { threshold: 0.4 })).rejects.toBeInstanceOf(
        BadRequestException,
      );
      await expect(sut.getUnassignedGroups(authStub.admin, newUuid(), { threshold: 0.4 })).rejects.toBeInstanceOf(
        BadRequestException,
      );
    });

    it('should return the group and who it looks like', async () => {
      const alice = person({ name: 'Alice', centroid: centroid(1, 0, 0) });
      const bob = person({ name: 'Bob', centroid: centroid(0, 1, 0.2) });
      const stranger = [face(centroid(0, 1, 0)), face(centroid(0, 1, 0.1)), face(centroid(0, 1, -0.1))];
      mocks.person.getCentroids.mockResolvedValue([alice, bob]);
      mocks.person.getFaceEmbeddings.mockResolvedValue([face(centroid(0, 0, 1)), ...stranger]);

      const { groups } = await sut.getUnassignedGroups(authStub.admin, stranger[2].id, { threshold: 0.4 });

      expect(groups).toHaveLength(1);
      expect(groups[0].faces.map(({ id }) => id).toSorted()).toEqual(stranger.map(({ id }) => id).toSorted());
      expect(groups[0].closestPerson).toEqual({ id: bob.personGroupId, name: 'Bob', distance: expect.any(Number) });
    });
  });

  describe('getGroups', () => {
    it('should require access to the person', async () => {
      await expect(sut.getGroups(authStub.admin, newUuid(), { threshold: 0.4 })).rejects.toBeInstanceOf(
        BadRequestException,
      );
      expect(mocks.person.getFaceEmbeddings).not.toHaveBeenCalled();
    });

    it('should return no groups for a person without faces', async () => {
      const alice = person({ centroid: centroid(1, 0, 0) });
      allowAccess(alice.personGroupId);
      mocks.person.getFaceEmbeddings.mockResolvedValue([]);

      await expect(sut.getGroups(authStub.admin, alice.personGroupId, { threshold: 0.4 })).resolves.toEqual({
        threshold: 0.4,
        truncated: false,
        groups: [],
      });
    });

    it('should group faces and suggest who a group looks like', async () => {
      const alice = person({ name: 'Alice', centroid: centroid(1, 0.2, 0) });
      const bob = person({ name: 'Bob', centroid: centroid(0, 0, 1) });
      const carol = person({ name: 'Carol', centroid: centroid(0, 1, 0) });
      allowAccess(alice.personGroupId);
      mocks.person.getCentroids.mockResolvedValue([alice, bob, carol]);

      const assetId = newUuid();
      const aliceFaces = [
        face(centroid(1, 0, 0), assetId),
        face(centroid(1, 0.1, 0), assetId),
        face(centroid(1, 0, 0.1)),
      ];
      const bobFaces = [face(centroid(0, 0.1, 1)), face(centroid(0.1, 0, 1))];
      mocks.person.getFaceEmbeddings.mockResolvedValue([bobFaces[0], ...aliceFaces, bobFaces[1]]);

      const { groups, truncated } = await sut.getGroups(authStub.admin, alice.personGroupId, { threshold: 0.4 });

      expect(truncated).toBe(false);
      expect(groups).toHaveLength(2);
      expect(groups[0]).toEqual(expect.objectContaining({ assetCount: 2, distanceToMain: 0, closestPerson: null }));
      expect(groups[0].faces.map(({ id }) => id).toSorted()).toEqual(aliceFaces.map(({ id }) => id).toSorted());
      expect(groups[1].closestPerson).toEqual({ id: bob.personGroupId, name: 'Bob', distance: expect.any(Number) });
      expect(groups[1].distanceToMain).toBeGreaterThan(0.5);
      expect(groups[1].faces.map(({ id }) => id).toSorted()).toEqual(bobFaces.map(({ id }) => id).toSorted());
      expect(groups[1].faces[0]).toEqual(
        expect.objectContaining({
          fileCreatedAt: '2026-01-01T00:00:00.000Z',
          imageWidth: 1000,
          boundingBoxX1: 10,
          distance: expect.any(Number),
        }),
      );
    });

    it('should reuse the people while regrouping', async () => {
      const alice = person({ centroid: centroid(1, 0, 0) });
      allowAccess(alice.personGroupId);
      mocks.person.getCentroids.mockResolvedValue([alice]);
      mocks.person.getFaceEmbeddings.mockResolvedValue([face(centroid(1, 0, 0))]);

      await sut.getGroups(authStub.admin, alice.personGroupId, { threshold: 0.4 });
      await sut.getGroups(authStub.admin, alice.personGroupId, { threshold: 0.3 });

      expect(mocks.person.getCentroids).toHaveBeenCalledTimes(1);
    });

    it('should report when a person has too many faces', async () => {
      const alice = person({ centroid: centroid(1, 0, 0) });
      allowAccess(alice.personGroupId);
      mocks.person.getCentroids.mockResolvedValue([alice]);
      mocks.person.getFaceEmbeddings.mockResolvedValue(Array.from({ length: 5001 }, () => face(centroid(1, 0, 0))));

      const { groups, truncated } = await sut.getGroups(authStub.admin, alice.personGroupId, { threshold: 0.4 });

      expect(truncated).toBe(true);
      expect(groups[0].faces).toHaveLength(5000);
    });
  });
});
