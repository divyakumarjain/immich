<script lang="ts">
  import FaceGroupsReview from '$lib/components/face-graph/FaceGroupsReview.svelte';
  import UserPageLayout from '$lib/components/layouts/UserPageLayout.svelte';
  import { Route } from '$lib/route';
  import { getPeopleThumbnailUrl } from '$lib/utils';
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import { handleError } from '$lib/utils/handle-error';
  import { getFaceGroups, reassignFaces, updatePerson, type FaceGroupFaceDto } from '@immich/sdk';
  import { Button, Input, Text, toastManager } from '@immich/ui';
  import { mdiArrowLeft, mdiFaceManOutline } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import type { PageData } from './$types';

  interface Props {
    data: PageData;
  }

  let { data }: Props = $props();

  let person = $derived(data.person);
  let name = $derived(data.person.name);
  let isSaving = $state(false);

  const onRename = async (event: SubmitEvent) => {
    event.preventDefault();
    const newName = name.trim();
    if (newName === person.name) {
      return;
    }

    isSaving = true;
    try {
      person = await updatePerson({ id: person.id, personUpdateDto: { name: newName } });
      name = person.name;
      faceGraphManager.updateNode(person.id, { name: person.name });
      toastManager.primary($t('change_name_successfully'));
    } catch (error) {
      handleError(error, $t('errors.unable_to_save_name'));
    } finally {
      isSaving = false;
    }
  };

  const onMove = async (target: { id: string }, faces: FaceGroupFaceDto[]) => {
    const assetIds = [...new Set(faces.map(({ assetId }) => assetId))];
    await reassignFaces({
      id: target.id,
      assetFaceUpdateDto: { data: assetIds.map((assetId) => ({ assetId, personId: person.id })) },
    });
    return assetIds.length;
  };
</script>

<UserPageLayout title={person.name || data.meta.title}>
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
            <form onsubmit={onRename} class="flex items-center gap-2" autocomplete="off">
              <Input
                bind:value={name}
                size="small"
                placeholder={$t('add_a_name')}
                aria-label={$t('name')}
                disabled={isSaving}
              />
              <Button type="submit" size="small" disabled={isSaving || name.trim() === person.name}>
                {$t('save')}
              </Button>
            </form>
            <Text size="small" color="muted">{$t('face_graph_faces', { values: { count: faceCount } })}</Text>
          </div>
        </div>
      {/snippet}
    </FaceGroupsReview>
  {/key}
</UserPageLayout>
