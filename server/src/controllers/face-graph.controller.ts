import { Controller, Get, Param, Query } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import { Endpoint, HistoryBuilder } from 'src/decorators';
import type { AuthDto } from 'src/dtos/auth.dto';
import { FaceGraphDto, FaceGraphResponseDto, FaceGroupsDto, FaceGroupsResponseDto } from 'src/dtos/face-graph.dto';
import { ApiTag, Permission } from 'src/enum';
import { Auth, Authenticated } from 'src/middleware/auth.guard';
import { FaceGraphService } from 'src/services/face-graph.service';
import { UUIDParamDto } from 'src/validation';

@ApiTags(ApiTag.People)
@Controller('face-graph')
export class FaceGraphController {
  constructor(private service: FaceGraphService) {}

  @Get()
  @Authenticated({ permission: Permission.PersonRead })
  @Endpoint({
    summary: 'Retrieve the face graph',
    description:
      'Retrieve the people of the authenticated user as a graph, in which people with similar faces are linked and positioned close to each other.',
    history: new HistoryBuilder().added('v3.3.0').alpha('v3.3.0'),
  })
  getFaceGraph(@Auth() auth: AuthDto, @Query() dto: FaceGraphDto): Promise<FaceGraphResponseDto> {
    return this.service.getGraph(auth, dto);
  }

  @Get('people/:id/groups')
  @Authenticated({ permission: Permission.PersonRead })
  @Endpoint({
    summary: 'Retrieve the face groups of a person',
    description:
      'Retrieve the faces of a person grouped by similarity. Groups that are far from the largest group may belong to a different person.',
    history: new HistoryBuilder().added('v3.3.0').alpha('v3.3.0'),
  })
  getFaceGroups(
    @Auth() auth: AuthDto,
    @Param() { id }: UUIDParamDto,
    @Query() dto: FaceGroupsDto,
  ): Promise<FaceGroupsResponseDto> {
    return this.service.getGroups(auth, id, dto);
  }

  @Get('unassigned/:id/groups')
  @Authenticated({ permission: Permission.FaceRead })
  @Endpoint({
    summary: 'Retrieve the face groups of unassigned faces',
    description:
      'Retrieve a group of similar faces that are not assigned to a person, identified by the ID of one of its faces, split further by similarity.',
    history: new HistoryBuilder().added('v3.3.0').alpha('v3.3.0'),
  })
  getUnassignedFaceGroups(
    @Auth() auth: AuthDto,
    @Param() { id }: UUIDParamDto,
    @Query() dto: FaceGroupsDto,
  ): Promise<FaceGroupsResponseDto> {
    return this.service.getUnassignedGroups(auth, id, dto);
  }
}
