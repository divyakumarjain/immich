<script lang="ts">
  import { goto } from '$app/navigation';
  import FaceGroupsReview from '$lib/components/face-graph/FaceGroupsReview.svelte';
  import UserPageLayout from '$lib/components/layouts/UserPageLayout.svelte';
  import { Route } from '$lib/route';
  import { getUnassignedFaceGroups, reassignFacesById, type FaceGroupFaceDto } from '@immich/sdk';
  import { Button, Text } from '@immich/ui';
  import { mdiArrowLeft } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import type { PageData } from './$types';

  interface Props {
    data: PageData;
  }

  let { data }: Props = $props();

  const MOVE_CONCURRENCY = 4;

  // the group is identified by one of its faces, which may be moved away
  let faceId = $derived(data.faceId);
  let currentFaces = $derived(data.faceGroups.groups.flatMap((group) => group.faces));

  const onLoad = async (threshold: number) => {
    const faceGroups = await getUnassignedFaceGroups({ id: faceId, threshold });
    currentFaces = faceGroups.groups.flatMap((group) => group.faces);
    return faceGroups;
  };

  const onMove = async (target: { id: string }, faces: FaceGroupFaceDto[]) => {
    const remaining = currentFaces.find(({ id }) => !faces.some((face) => face.id === id));
    for (let i = 0; i < faces.length; i += MOVE_CONCURRENCY) {
      await Promise.all(
        faces.slice(i, i + MOVE_CONCURRENCY).map(({ id }) => reassignFacesById({ id: target.id, faceDto: { id } })),
      );
    }
    if (remaining) {
      faceId = remaining.id;
    }
    return new Set(faces.map(({ assetId }) => assetId)).size;
  };
</script>

<UserPageLayout title={data.meta.title}>
  {#snippet buttons()}
    <Button size="small" variant="ghost" color="secondary" leadingIcon={mdiArrowLeft} href={Route.faceGraphUtility()}>
      {$t('face_graph')}
    </Button>
  {/snippet}

  {#key data.faceId}
    <FaceGroupsReview
      faceGroups={data.faceGroups}
      alwaysSuggest
      {onLoad}
      {onMove}
      onEmpty={() => goto(Route.faceGraphUtility())}
    >
      {#snippet header({ faceCount })}
        <div>
          <p class="text-lg font-medium">{$t('face_graph_unassigned_faces')}</p>
          <Text size="small" color="muted">{$t('face_graph_faces', { values: { count: faceCount } })}</Text>
        </div>
      {/snippet}
    </FaceGroupsReview>
  {/key}
</UserPageLayout>
