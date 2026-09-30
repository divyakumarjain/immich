<script lang="ts">
  import FaceGroupsReview from '$lib/components/face-graph/FaceGroupsReview.svelte';
  import UserPageLayout from '$lib/components/layouts/UserPageLayout.svelte';
  import { Route } from '$lib/route';
  import { getPeopleThumbnailUrl } from '$lib/utils';
  import { getFaceGroups, reassignFaces, type FaceGroupFaceDto } from '@immich/sdk';
  import { Button, Text } from '@immich/ui';
  import { mdiArrowLeft, mdiFaceManOutline } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import type { PageData } from './$types';

  interface Props {
    data: PageData;
  }

  let { data }: Props = $props();

  const person = $derived(data.person);

  const onMove = async (target: { id: string }, faces: FaceGroupFaceDto[]) => {
    const assetIds = [...new Set(faces.map(({ assetId }) => assetId))];
    await reassignFaces({
      id: target.id,
      assetFaceUpdateDto: { data: assetIds.map((assetId) => ({ assetId, personId: person.id })) },
    });
    return assetIds.length;
  };
</script>

<UserPageLayout title={data.meta.title}>
  {#snippet buttons()}
    <div class="flex items-center gap-1">
      <Button size="small" variant="ghost" color="secondary" leadingIcon={mdiArrowLeft} href={Route.faceGraphUtility()}>
        {$t('face_graph')}
      </Button>
      <Button
        size="small"
        variant="ghost"
        color="secondary"
        leadingIcon={mdiFaceManOutline}
        href={Route.viewPerson(person)}
      >
        {$t('face_graph_view_person')}
      </Button>
    </div>
  {/snippet}

  {#key person.id}
    <FaceGroupsReview
      faceGroups={data.faceGroups}
      personId={person.id}
      onLoad={(threshold) => getFaceGroups({ id: person.id, threshold })}
      {onMove}
    >
      {#snippet header({ faceCount })}
        <div class="flex items-center gap-3">
          <img src={getPeopleThumbnailUrl(person)} alt={person.name} class="size-16 rounded-full object-cover" />
          <div>
            <p class="text-lg font-medium">{person.name || $t('add_a_name')}</p>
            <Text size="small" color="muted">{$t('face_graph_faces', { values: { count: faceCount } })}</Text>
          </div>
        </div>
      {/snippet}
    </FaceGroupsReview>
  {/key}
</UserPageLayout>
