<script lang="ts">
  import FaceGroupsReview from '$lib/components/face-graph/FaceGroupsReview.svelte';
  import UserPageLayout from '$lib/components/layouts/UserPageLayout.svelte';
  import { Route } from '$lib/route';
  import { getPeopleThumbnailUrl } from '$lib/utils';
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import { handleError } from '$lib/utils/handle-error';
  import { goto } from '$app/navigation';
  import PersonNameInput from '$lib/components/face-graph/PersonNameInput.svelte';
  import {
    getFaceGroups,
    mergePeople,
    reassignFaces,
    updatePerson,
    type FaceGroupFaceDto,
    type PersonResponseDto,
  } from '@immich/sdk';
  import { Button, modalManager, Text, toastManager } from '@immich/ui';
  import { mdiArrowLeft, mdiFaceManOutline } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import type { PageData } from './$types';

  interface Props {
    data: PageData;
  }

  let { data }: Props = $props();

  let person = $derived(data.person);
  let isSaving = $state(false);

  /** this person turns out to be someone who already exists: merge and continue with that person */
  const onMergeInto = async (target: PersonResponseDto) => {
    const isConfirmed = await modalManager.showDialog({
      prompt: $t('face_graph_merge_confirm', { values: { count: 1, name: target.name || null } }),
    });
    if (!isConfirmed) {
      return false;
    }

    isSaving = true;
    try {
      await mergePeople({ mergePersonDto: { ids: [target.id, person.id] } });
      faceGraphManager.invalidate();
      toastManager.primary($t('merge_people_successfully'));
      await goto(Route.faceGraphPerson(target));
      return true;
    } catch (error) {
      handleError(error, $t('cannot_merge_people'));
      return false;
    } finally {
      isSaving = false;
    }
  };

  const onRename = async (newName: string) => {
    isSaving = true;
    try {
      person = await updatePerson({ id: person.id, personUpdateDto: { name: newName } });
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
            <PersonNameInput
              personId={person.id}
              name={person.name}
              disabled={isSaving}
              onSave={onRename}
              onPick={onMergeInto}
            />
            <Text size="small" color="muted">{$t('face_graph_faces', { values: { count: faceCount } })}</Text>
          </div>
        </div>
      {/snippet}
    </FaceGroupsReview>
  {/key}
</UserPageLayout>
