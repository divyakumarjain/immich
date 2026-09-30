import { getUnassignedFaceGroups } from '@immich/sdk';
import { authenticate } from '$lib/utils/auth';
import { getFormatter } from '$lib/utils/i18n';
import type { PageLoad } from './$types';

export const load = (async ({ params, url }) => {
  await authenticate(url);

  const faceGroups = await getUnassignedFaceGroups({ id: params.faceId });
  const $t = await getFormatter();

  return {
    faceId: params.faceId,
    faceGroups,
    meta: {
      title: $t('face_graph_unassigned_faces'),
    },
  };
}) satisfies PageLoad;
