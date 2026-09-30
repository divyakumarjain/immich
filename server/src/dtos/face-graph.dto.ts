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

export class FaceGraphDto extends createZodDto(FaceGraphSchema) {}
export class FaceGraphResponseDto extends createZodDto(FaceGraphResponseSchema) {}
