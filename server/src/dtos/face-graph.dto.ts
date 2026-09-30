import { createZodDto } from 'nestjs-zod';
import z from 'zod';
import { stringToBool } from 'src/validation.js';

export enum FaceGraphNodeKind {
  Person = 'person',
  Unassigned = 'unassigned',
}

const FaceGraphNodeKindSchema = z
  .enum(FaceGraphNodeKind)
  .describe('Whether the node is a person or a group of faces that are not assigned to a person')
  .meta({ id: 'FaceGraphNodeKind' });

const FaceGraphSchema = z
  .object({
    minFaces: z.coerce.number().int().min(1).default(1).describe('Only include people with at least this many faces'),
    withHidden: stringToBool.optional().describe('Include hidden people'),
    neighbors: z.coerce.number().int().min(1).max(20).default(5).describe('Maximum number of similar people to link'),
    maxDistance: z.coerce
      .number()
      .min(0)
      .max(2)
      .meta({ format: 'double' })
      .optional()
      .describe('Maximum distance between linked people, defaults to the facial recognition distance'),
  })
  .meta({ id: 'FaceGraphDto' });

const FaceGraphNodeSchema = z
  .object({
    id: z.uuidv4().describe('Person ID'),
    kind: FaceGraphNodeKindSchema,
    name: z.string().describe('Person name'),
    isHidden: z.boolean().describe('Is hidden'),
    isFavorite: z.boolean().describe('Is favorite'),
    // TODO: use `isoDatetimeToDate` when using `ZodSerializerDto` on the controllers.
    updatedAt: z.string().meta({ format: 'date-time' }).describe('Last update date'),
    assetCount: z.int().min(0).describe('Number of assets the person appears in'),
    faceCount: z.int().min(0).describe('Number of faces assigned to the person'),
    x: z.number().meta({ format: 'double' }).describe('Suggested horizontal position, between -1 and 1'),
    y: z.number().meta({ format: 'double' }).describe('Suggested vertical position, between -1 and 1'),
  })
  .meta({ id: 'FaceGraphNodeDto' });

const FaceGraphEdgeSchema = z
  .object({
    source: z.uuidv4().describe('Node ID'),
    target: z.uuidv4().describe('Node ID'),
    distance: z.number().meta({ format: 'double' }).describe('Distance between the two nodes, lower is more similar'),
  })
  .meta({ id: 'FaceGraphEdgeDto' });

const FaceGraphResponseSchema = z
  .object({
    nodes: z.array(FaceGraphNodeSchema),
    edges: z.array(FaceGraphEdgeSchema).describe('Links between similar nodes'),
  })
  .meta({ id: 'FaceGraphResponseDto' });

const FaceGroupsSchema = z
  .object({
    threshold: z.coerce
      .number()
      .min(0.05)
      .max(1)
      .default(0.4)
      .meta({ format: 'double' })
      .describe('Maximum distance of a face to its group, lower values create more groups'),
  })
  .meta({ id: 'FaceGroupsDto' });

const FaceGroupFaceSchema = z
  .object({
    id: z.uuidv4().describe('Face ID'),
    assetId: z.uuidv4().describe('Asset ID'),
    // TODO: use `isoDatetimeToDate` when using `ZodSerializerDto` on the controllers.
    fileCreatedAt: z.string().meta({ format: 'date-time' }).describe('Date the asset was taken'),
    distance: z.number().meta({ format: 'double' }).describe('Distance to the largest group, lower is more similar'),
    imageHeight: z.int().min(0).describe('Image height in pixels'),
    imageWidth: z.int().min(0).describe('Image width in pixels'),
    boundingBoxX1: z.int().describe('Bounding box X1 coordinate'),
    boundingBoxX2: z.int().describe('Bounding box X2 coordinate'),
    boundingBoxY1: z.int().describe('Bounding box Y1 coordinate'),
    boundingBoxY2: z.int().describe('Bounding box Y2 coordinate'),
  })
  .meta({ id: 'FaceGroupFaceDto' });

const FaceGroupClosestPersonSchema = z
  .object({
    id: z.uuidv4().describe('Person ID'),
    name: z.string().describe('Person name'),
    distance: z.number().meta({ format: 'double' }).describe('Distance to the person, lower is more similar'),
  })
  .meta({ id: 'FaceGroupClosestPersonDto' });

const FaceGroupSchema = z
  .object({
    id: z.uuidv4().describe('ID of the most representative face of the group'),
    assetCount: z.int().min(0).describe('Number of assets in the group'),
    distanceToMain: z
      .number()
      .meta({ format: 'double' })
      .describe('Distance to the largest group, lower is more similar'),
    closestPerson: FaceGroupClosestPersonSchema.nullable().describe('The other person that looks most like the group'),
    faces: z.array(FaceGroupFaceSchema).describe('Faces of the group, the most representative first'),
  })
  .meta({ id: 'FaceGroupDto' });

const FaceGroupsResponseSchema = z
  .object({
    threshold: z.number().meta({ format: 'double' }).describe('Threshold used to create the groups'),
    truncated: z.boolean().describe('Whether only the most recent faces of the person were grouped'),
    groups: z.array(FaceGroupSchema).describe('Groups of similar faces, largest first'),
  })
  .meta({ id: 'FaceGroupsResponseDto' });

export class FaceGraphDto extends createZodDto(FaceGraphSchema) {}
export class FaceGroupsDto extends createZodDto(FaceGroupsSchema) {}
export class FaceGroupsResponseDto extends createZodDto(FaceGroupsResponseSchema) {}
export class FaceGraphResponseDto extends createZodDto(FaceGraphResponseSchema) {}
