<script lang="ts">
  import FaceGroupCard from '$lib/components/face-graph/FaceGroupCard.svelte';
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import PeoplePickerModal from '$lib/modals/PeoplePickerModal.svelte';
  import { handleError } from '$lib/utils/handle-error';
  import {
    createPerson,
    deleteFace,
    type FaceGroupDto,
    type FaceGroupFaceDto,
    type FaceGroupsResponseDto,
    type PersonResponseDto,
  } from '@immich/sdk';
  import { Button, Label, modalManager, Text, toastManager } from '@immich/ui';
  import { mdiAccountArrowRightOutline, mdiAccountPlusOutline, mdiClose } from '@mdi/js';
  import type { Snippet } from 'svelte';
  import { t } from 'svelte-i18n';
  import { SvelteSet } from 'svelte/reactivity';

  type Target = Pick<PersonResponseDto, 'id' | 'name'>;

  type Props = {
    faceGroups: FaceGroupsResponseDto;
    /** the person the faces belong to, not offered as a target */
    personId?: string;
    /** suggest a person for every group, for faces that do not belong to anyone yet */
    alwaysSuggest?: boolean;
    onLoad: (threshold: number) => Promise<FaceGroupsResponseDto>;
    /** moves the faces to the person and returns the number of moved photos */
    onMove: (target: Target, faces: FaceGroupFaceDto[]) => Promise<number>;
    /** called when there is nothing left to review */
    onEmpty?: () => void;
    header: Snippet<[{ faceCount: number }]>;
  };

  let {
    faceGroups: initialFaceGroups,
    personId,
    alwaysSuggest = false,
    onLoad,
    onMove,
    onEmpty,
    header,
  }: Props = $props();

  const DELETE_CONCURRENCY = 4;

  let faceGroups = $derived(initialFaceGroups);
  let threshold = $derived(initialFaceGroups.threshold);
  let isBusy = $state(false);
  const selection = new SvelteSet<string>();

  const faces = $derived(faceGroups.groups.flatMap((group) => group.faces));
  const selectedFaces = $derived(faces.filter(({ id }) => selection.has(id)));
  const mainGroupId = $derived(faceGroups.groups.at(0)?.id);

  const reload = async () => {
    try {
      faceGroups = await onLoad(threshold);
    } catch (error) {
      // the last faces were moved or removed
      faceGroups = { threshold, truncated: false, groups: [] };
      if (!onEmpty) {
        handleError(error, $t('errors.unable_to_load_face_groups'));
      }
    }

    const ids = new Set(faceGroups.groups.flatMap((group) => group.faces.map(({ id }) => id)));
    for (const id of selection) {
      if (!ids.has(id)) {
        selection.delete(id);
      }
    }
    if (ids.size === 0) {
      onEmpty?.();
    }
  };

  const afterChange = async () => {
    selection.clear();
    // the people in the graph changed size, or are new
    faceGraphManager.invalidate();
    await reload();
  };

  const moveToExisting = async (target: Target, faces: FaceGroupFaceDto[]) => {
    isBusy = true;
    try {
      const count = await onMove(target, faces);
      toastManager.primary(
        $t('reassigned_assets_to_existing_person', { values: { count, name: target.name || null } }),
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
    isBusy = true;
    try {
      const target = await createPerson({ personCreateDto: {} });
      const count = await onMove(target, selectedFaces);
      toastManager.primary($t('reassigned_assets_to_new_person', { values: { count } }));
      await afterChange();
    } catch (error) {
      handleError(error, $t('errors.unable_to_reassign_assets_new_person'));
    } finally {
      isBusy = false;
    }
  };

  const onMoveToExisting = async () => {
    const people = await modalManager.show(PeoplePickerModal, { excludedIds: personId ? [personId] : [] });
    if (people?.[0]) {
      await moveToExisting(people[0], selectedFaces);
    }
  };

  const onMoveToClosest = async (group: FaceGroupDto) => {
    if (group.closestPerson) {
      await moveToExisting(group.closestPerson, group.faces);
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

<div class="mx-auto flex max-w-6xl flex-col gap-4 p-2 pb-24">
  <div class="flex flex-wrap items-center gap-x-8 gap-y-3 text-dark">
    {@render header({ faceCount: faces.length })}

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
    <FaceGroupCard
      {group}
      isMain={group.id === mainGroupId}
      {alwaysSuggest}
      {selection}
      disabled={isBusy}
      {onMoveToClosest}
    />
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
