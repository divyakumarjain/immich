import { getFaceGroups, getPerson } from '@immich/sdk';
import { authenticate } from '$lib/utils/auth';
import { getFormatter } from '$lib/utils/i18n';
import type { PageLoad } from './$types';

export const load = (async ({ params, url }) => {
  await authenticate(url);

  const [person, faceGroups] = await Promise.all([
    getPerson({ id: params.personId }),
    getFaceGroups({ id: params.personId }),
  ]);
  const $t = await getFormatter();

  return {
    person,
    faceGroups,
    meta: {
      title: person.name || $t('person'),
    },
  };
}) satisfies PageLoad;
