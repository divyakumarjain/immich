<script lang="ts">
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import { Route } from '$lib/route';
  import FaceGraphNodeThumbnail from '$lib/components/face-graph/FaceGraphNodeThumbnail.svelte';
  import { findMergeTarget, findNodeByName, isUnassigned } from '$lib/utils/face-graph';
  import { handleError } from '$lib/utils/handle-error';
  import { mergePeople, updatePerson, type FaceGraphNodeDto } from '@immich/sdk';
  import PersonNameInput from '$lib/components/face-graph/PersonNameInput.svelte';
  import { Button, modalManager, Text, toastManager } from '@immich/ui';
  import { mdiCallMerge, mdiEyeOffOutline, mdiEyeOutline, mdiFaceRecognition } from '@mdi/js';
  import { t } from 'svelte-i18n';

  const selectedNodes = $derived(
    faceGraphManager.selectedIds
      .map((id) => faceGraphManager.nodeById.get(id))
      .filter((node): node is FaceGraphNodeDto => !!node),
  );
  const node = $derived(selectedNodes.length === 1 ? selectedNodes[0] : undefined);

  let isBusy = $state(false);
  let nameInput = $state<HTMLInputElement | null>(null);

  export const focusName = () => nameInput?.focus();

  const merge = async (target: Pick<FaceGraphNodeDto, 'id' | 'name'>, others: FaceGraphNodeDto[]) => {
    const isConfirmed = await modalManager.showDialog({
      prompt: $t('face_graph_merge_confirm', { values: { count: others.length, name: target.name || null } }),
    });
    if (!isConfirmed) {
      return false;
    }

    await mergePeople({ mergePersonDto: { ids: [target.id, ...others.map(({ id }) => id)] } });
    faceGraphManager.mergeNodes(
      target.id,
      others.map(({ id }) => id),
    );
    toastManager.primary($t('merge_people_successfully'));
    return true;
  };

  const onMerge = async () => {
    const target = findMergeTarget(selectedNodes);
    if (!target || selectedNodes.some((node) => isUnassigned(node))) {
      return;
    }

    isBusy = true;
    try {
      await merge(
        target,
        selectedNodes.filter(({ id }) => id !== target.id),
      );
    } catch (error) {
      handleError(error, $t('cannot_merge_people'));
    } finally {
      isBusy = false;
    }
  };

  /** merges the selected person into an existing one, e.g. one picked from the name suggestions */
  const mergeInto = async (target: Pick<FaceGraphNodeDto, 'id' | 'name'>) => {
    if (!node) {
      return false;
    }

    isBusy = true;
    try {
      if (await merge(target, [node])) {
        // the target may not be in the graph, e.g. when it has no visible photos
        if (faceGraphManager.nodeById.has(target.id)) {
          faceGraphManager.focusOn(target.id);
        }
        return true;
      }
    } catch (error) {
      handleError(error, $t('cannot_merge_people'));
    } finally {
      isBusy = false;
    }
    return false;
  };

  const onRename = async (newName: string) => {
    if (!node) {
      return;
    }

    isBusy = true;
    try {
      const existing = findNodeByName(faceGraphManager.nodes, newName, node.id);
      if (existing && (await merge(existing, [node]))) {
        faceGraphManager.focusOn(existing.id);
        return;
      }

      const person = await updatePerson({ id: node.id, personUpdateDto: { name: newName } });
      faceGraphManager.updateNode(node.id, { name: person.name });
      toastManager.primary($t('change_name_successfully'));
    } catch (error) {
      handleError(error, $t('errors.unable_to_save_name'));
    } finally {
      isBusy = false;
    }
  };

  const onToggleHidden = async () => {
    if (!node) {
      return;
    }

    isBusy = true;
    try {
      const person = await updatePerson({ id: node.id, personUpdateDto: { isHidden: !node.isHidden } });
      faceGraphManager.updateNode(node.id, { isHidden: person.isHidden });
      toastManager.primary($t('changed_visibility_successfully'));
    } catch (error) {
      handleError(error, $t('errors.unable_to_hide_person'));
    } finally {
      isBusy = false;
    }
  };
</script>

{#if node}
  <section class="flex flex-col gap-3 border-b p-3">
    <div class="flex items-center gap-3">
      <FaceGraphNodeThumbnail {node} size={56} />
      <Text size="small" color="muted">
        {$t('face_graph_photos_and_faces', { values: { photos: node.assetCount, faces: node.faceCount } })}
      </Text>
    </div>

    {#if isUnassigned(node)}
      <Text size="small">{$t('face_graph_unassigned_hint')}</Text>
      <Button size="small" leadingIcon={mdiFaceRecognition} href={Route.faceGraphUnassigned(node)}>
        {$t('face_graph_review_faces')}
      </Button>
    {:else}
      {#key node.id}
        <PersonNameInput
          bind:ref={nameInput}
          personId={node.id}
          name={node.name}
          disabled={isBusy}
          onSave={onRename}
          onPick={mergeInto}
        />
      {/key}

      <div class="flex flex-wrap gap-2">
        <Button size="small" color="secondary" leadingIcon={mdiFaceRecognition} href={Route.faceGraphPerson(node)}>
          {$t('face_graph_review_faces')}
        </Button>
        <Button
          size="small"
          color="secondary"
          variant="ghost"
          leadingIcon={node.isHidden ? mdiEyeOutline : mdiEyeOffOutline}
          disabled={isBusy}
          onclick={onToggleHidden}
        >
          {node.isHidden ? $t('unhide_person') : $t('hide_person')}
        </Button>
      </div>
    {/if}
  </section>
{:else if selectedNodes.length > 1}
  <section class="flex flex-col gap-3 border-b p-3">
    <div class="flex flex-wrap gap-1">
      {#each selectedNodes as selectedNode (selectedNode.id)}
        <FaceGraphNodeThumbnail node={selectedNode} size={40} />
      {/each}
    </div>
    <Button
      size="small"
      leadingIcon={mdiCallMerge}
      loading={isBusy}
      disabled={selectedNodes.some((node) => isUnassigned(node))}
      onclick={onMerge}
    >
      {$t('face_graph_merge_people', { values: { count: selectedNodes.length } })}
    </Button>
    <Text size="tiny" color="muted">{$t('face_graph_merge_hint')}</Text>
  </section>
{/if}
