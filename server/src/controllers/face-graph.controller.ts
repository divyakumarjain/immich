import { Controller, Get, Query } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import type { AuthDto } from 'src/dtos/auth.dto.js';
import { Endpoint, HistoryBuilder } from 'src/decorators.js';
import { FaceGraphDto, FaceGraphResponseDto } from 'src/dtos/face-graph.dto.js';
import { ApiTag, Permission } from 'src/enum.js';
import { Auth, Authenticated } from 'src/middleware/auth.guard.js';
import { FaceGraphService } from 'src/services/face-graph.service.js';

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
}
