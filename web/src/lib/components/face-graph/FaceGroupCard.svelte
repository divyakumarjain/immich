<script lang="ts">
  import { Route } from '$lib/route';
  import { looksLikeSomeoneElse } from '$lib/utils/face-graph';
  import type { FaceGroupDto } from '@immich/sdk';
  import { Button, Checkbox, Icon, Label, Text } from '@immich/ui';
  import { mdiAccountArrowRightOutline, mdiAlertOutline, mdiCheckCircle, mdiOpenInNew } from '@mdi/js';
  import { t } from 'svelte-i18n';
  import type { SvelteSet } from 'svelte/reactivity';
  import FaceCrop from '$lib/components/face-graph/FaceCrop.svelte';

  type Props = {
    group: FaceGroupDto;
    isMain: boolean;
    /** suggest a person for every group, for faces that do not belong to anyone yet */
    alwaysSuggest?: boolean;
    selection: SvelteSet<string>;
    disabled: boolean;
    onMoveToClosest: (group: FaceGroupDto) => void;
  };

  let { group, isMain, alwaysSuggest = false, selection, disabled, onMoveToClosest }: Props = $props();

  const FACE_SIZE = 96;
  const INITIAL_FACES = 24;
  const SUGGESTION_MAX_DISTANCE = 0.6;

  let showAll = $state(false);

  const faces = $derived(showAll ? group.faces : group.faces.slice(0, INITIAL_FACES));
  const selectedCount = $derived(group.faces.filter(({ id }) => selection.has(id)).length);
  const isSuspicious = $derived(looksLikeSomeoneElse(group));
  const suggestion = $derived(
    isSuspicious || (alwaysSuggest && (group.closestPerson?.distance ?? Infinity) <= SUGGESTION_MAX_DISTANCE)
      ? group.closestPerson
      : null,
  );
  const distanceClass = $derived(
    group.distanceToMain > 0.5
      ? 'bg-danger/15 text-danger'
      : group.distanceToMain > 0.35
        ? 'bg-warning/15'
        : 'bg-subtle',
  );

  const toggleGroup = (checked: boolean) => {
    for (const { id } of group.faces) {
      if (checked) {
        selection.add(id);
      } else {
        selection.delete(id);
      }
    }
  };

  const toggleFace = (id: string) => {
    if (selection.has(id)) {
      selection.delete(id);
    } else {
      selection.add(id);
    }
  };
</script>

<section class="rounded-2xl border p-4 text-dark {isSuspicious ? 'border-warning' : ''}">
  <header class="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
    <div class="flex items-center gap-2">
      <Checkbox
        id="face-group-{group.id}"
        {disabled}
        checked={selectedCount === group.faces.length}
        indeterminate={selectedCount > 0 && selectedCount < group.faces.length}
        onCheckedChange={toggleGroup}
      />
      <Label for="face-group-{group.id}" class="font-medium">
        {isMain
          ? $t('face_graph_main_group')
          : $t('face_graph_photos_and_faces', { values: { photos: group.assetCount, faces: group.faces.length } })}
      </Label>
    </div>

    {#if isMain}
      <Text size="small" color="muted">
        {$t('face_graph_photos_and_faces', { values: { photos: group.assetCount, faces: group.faces.length } })}
      </Text>
    {:else}
      <span class="rounded-full px-2 py-0.5 text-xs tabular-nums {distanceClass}">
        {$t('face_graph_distance', { values: { distance: group.distanceToMain.toFixed(2) } })}
      </span>
    {/if}

    {#if suggestion}
      <span class="flex items-center gap-1 text-sm">
        <Icon icon={mdiAlertOutline} size="18" class="text-warning" />
        {$t('face_graph_looks_like', { values: { name: suggestion.name || null } })}
        <span class="text-xs tabular-nums opacity-70">({suggestion.distance.toFixed(2)})</span>
      </span>
      <Button
        size="small"
        variant="outline"
        leadingIcon={mdiAccountArrowRightOutline}
        {disabled}
        onclick={() => onMoveToClosest(group)}
      >
        {$t('face_graph_move_group', { values: { name: suggestion.name || null } })}
      </Button>
    {/if}
  </header>

  <div class="flex flex-wrap gap-2">
    {#each faces as face (face.id)}
      {@const isSelected = selection.has(face.id)}
      <div class="group relative">
        <button
          type="button"
          {disabled}
          aria-pressed={isSelected}
          aria-label={$t('select')}
          class="block overflow-hidden rounded-lg outline-offset-2 {isSelected ? 'outline-3 outline-primary' : ''}"
          onclick={() => toggleFace(face.id)}
        >
          <FaceCrop {face} size={FACE_SIZE} />
        </button>
        {#if isSelected}
          <Icon icon={mdiCheckCircle} size="20" class="pointer-events-none absolute inset-s-1 top-1 text-primary" />
        {/if}
        <a
          href={Route.viewAsset({ id: face.assetId })}
          target="_blank"
          rel="noreferrer"
          title={$t('face_graph_open_photo')}
          aria-label={$t('face_graph_open_photo')}
          class="absolute inset-e-1 top-1 rounded-full bg-black/60 p-1 text-white opacity-0 group-hover:opacity-100 focus:opacity-100"
        >
          <Icon icon={mdiOpenInNew} size="14" />
        </a>
      </div>
    {/each}
  </div>

  {#if !showAll && group.faces.length > INITIAL_FACES}
    <Button size="small" variant="ghost" class="mt-3" onclick={() => (showAll = true)}>
      {$t('face_graph_show_all_faces', { values: { count: group.faces.length } })}
    </Button>
  {/if}
</section>
