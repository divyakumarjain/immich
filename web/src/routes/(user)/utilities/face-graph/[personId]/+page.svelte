<script lang="ts">
  import UserPageLayout from '$lib/components/layouts/UserPageLayout.svelte';
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import PeoplePickerModal from '$lib/modals/PeoplePickerModal.svelte';
  import { Route } from '$lib/route';
  import { getPeopleThumbnailUrl } from '$lib/utils';
  import { handleError } from '$lib/utils/handle-error';
  import {
    createPerson,
    deleteFace,
    getFaceGroups,
    reassignFaces,
    type FaceGroupDto,
    type PersonResponseDto,
  } from '@immich/sdk';
  import { Button, Label, modalManager, Text, toastManager } from '@immich/ui';
  import {
    mdiAccountArrowRightOutline,
    mdiAccountPlusOutline,
    mdiArrowLeft,
    mdiClose,
    mdiFaceManOutline,
  } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import { SvelteSet } from 'svelte/reactivity';
  import type { PageData } from './$types';
  import FaceGroupCard from './FaceGroupCard.svelte';

  interface Props {
    data: PageData;
  }

  let { data }: Props = $props();

  const DELETE_CONCURRENCY = 4;

  const person = $derived(data.person);
  let faceGroups = $derived(data.faceGroups);
  let threshold = $derived(data.faceGroups.threshold);
  let isBusy = $state(false);
  const selection = new SvelteSet<string>();

  const faces = $derived(faceGroups.groups.flatMap((group) => group.faces));
  const selectedFaces = $derived(faces.filter(({ id }) => selection.has(id)));
  const mainGroupId = $derived(faceGroups.groups.at(0)?.id);

  const reload = async () => {
    try {
      faceGroups = await getFaceGroups({ id: person.id, threshold });
      const ids = new Set(faceGroups.groups.flatMap((group) => group.faces.map(({ id }) => id)));
      for (const id of selection) {
        if (!ids.has(id)) {
          selection.delete(id);
        }
      }
    } catch (error) {
      handleError(error, $t('errors.unable_to_load_face_groups'));
    }
  };

  const afterChange = async () => {
    selection.clear();
    // the people in the graph changed size, or are new
    faceGraphManager.invalidate();
    await reload();
  };

  const getAssetIds = (faces: { assetId: string }[]) => [...new Set(faces.map(({ assetId }) => assetId))];

  const moveTo = async (target: Pick<PersonResponseDto, 'id' | 'name'>, faces: { assetId: string }[]) => {
    const assetIds = getAssetIds(faces);
    isBusy = true;
    try {
      await reassignFaces({
        id: target.id,
        assetFaceUpdateDto: { data: assetIds.map((assetId) => ({ assetId, personId: person.id })) },
      });
      toastManager.primary(
        $t('reassigned_assets_to_existing_person', { values: { count: assetIds.length, name: target.name || null } }),
      );
      await afterChange();
    } catch (error) {
      handleError(
        error,
        $t('errors.unable_to_reassign_assets_existing_person', { values: { name: target.name || null } }),
      );
    } finally {
      isBusy = false;
    }
  };

  const onMoveToNew = async () => {
    const assetIds = getAssetIds(selectedFaces);
    isBusy = true;
    try {
      const target = await createPerson({ personCreateDto: {} });
      await reassignFaces({
        id: target.id,
        assetFaceUpdateDto: { data: assetIds.map((assetId) => ({ assetId, personId: person.id })) },
      });
      toastManager.primary($t('reassigned_assets_to_new_person', { values: { count: assetIds.length } }));
      await afterChange();
    } catch (error) {
      handleError(error, $t('errors.unable_to_reassign_assets_new_person'));
    } finally {
      isBusy = false;
    }
  };

  const onMoveToExisting = async () => {
    const people = await modalManager.show(PeoplePickerModal, { excludedIds: [person.id] });
    if (people?.[0]) {
      await moveTo(people[0], selectedFaces);
    }
  };

  const onMoveToClosest = async (group: FaceGroupDto) => {
    if (group.closestPerson) {
      await moveTo(group.closestPerson, group.faces);
    }
  };

  const onRemove = async () => {
    const ids = selectedFaces.map(({ id }) => id);
    const isConfirmed = await modalManager.showDialog({
      prompt: $t('face_graph_remove_faces_confirm', { values: { count: ids.length } }),
    });
    if (!isConfirmed) {
      return;
    }

    isBusy = true;
    try {
      for (let i = 0; i < ids.length; i += DELETE_CONCURRENCY) {
        await Promise.all(
          ids.slice(i, i + DELETE_CONCURRENCY).map((id) => deleteFace({ id, assetFaceDeleteDto: { force: false } })),
        );
      }
      toastManager.primary($t('face_graph_faces_removed', { values: { count: ids.length } }));
    } catch (error) {
      handleError(error, $t('error_delete_face'));
    } finally {
      isBusy = false;
      await afterChange();
    }
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

  <div class="mx-auto flex max-w-6xl flex-col gap-4 p-2 pb-24">
    <div class="flex flex-wrap items-center gap-x-8 gap-y-3 text-dark">
      <div class="flex items-center gap-3">
        <img src={getPeopleThumbnailUrl(person)} alt={person.name} class="size-16 rounded-full object-cover" />
        <div>
          <p class="text-lg font-medium">{person.name || $t('add_a_name')}</p>
          <Text size="small" color="muted">{$t('face_graph_faces', { values: { count: faces.length } })}</Text>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <Label for="face-graph-threshold">{$t('face_graph_more_groups')}</Label>
        <input
          id="face-graph-threshold"
          type="range"
          min="0.2"
          max="0.6"
          step="0.05"
          bind:value={threshold}
          onchange={reload}
          class="w-40 accent-primary"
        />
        <Text size="small">{$t('face_graph_fewer_groups')}</Text>
      </div>
    </div>

    <Text size="small" color="muted">{$t('face_graph_groups_description')}</Text>

    {#if faceGroups.truncated}
      <Text size="small" color="warning">{$t('face_graph_truncated', { values: { count: faces.length } })}</Text>
    {/if}

    {#each faceGroups.groups as group (group.id)}
      <FaceGroupCard {group} isMain={group.id === mainGroupId} {selection} disabled={isBusy} {onMoveToClosest} />
    {:else}
      <p class="py-16 text-center text-dark">{$t('face_graph_no_faces')}</p>
    {/each}
  </div>

  {#if selection.size > 0}
    <div class="fixed inset-x-0 bottom-4 z-10 flex justify-center px-4">
      <div class="flex flex-wrap items-center gap-2 rounded-2xl border bg-light px-4 py-3 text-dark shadow-lg">
        <Text fontWeight="medium" class="pe-2">{$t('selected_count', { values: { count: selection.size } })}</Text>
        <Button size="small" leadingIcon={mdiAccountPlusOutline} loading={isBusy} onclick={onMoveToNew}>
          {$t('face_graph_move_to_new_person')}
        </Button>
        <Button
          size="small"
          color="secondary"
          leadingIcon={mdiAccountArrowRightOutline}
          disabled={isBusy}
          onclick={onMoveToExisting}
        >
          {$t('face_graph_move_to_existing_person')}
        </Button>
        <Button size="small" color="danger" variant="outline" disabled={isBusy} onclick={onRemove}>
          {$t('face_graph_remove_faces')}
        </Button>
        <Button size="small" variant="ghost" color="secondary" leadingIcon={mdiClose} onclick={() => selection.clear()}>
          {$t('clear_all')}
        </Button>
      </div>
    </div>
  {/if}
</UserPageLayout>
