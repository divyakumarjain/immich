<script lang="ts">
  import { goto } from '$app/navigation';
  import { shortcuts } from '$lib/actions/shortcut';
  import FaceGraphCanvas from '$lib/components/face-graph/FaceGraphCanvas.svelte';
  import UserPageLayout from '$lib/components/layouts/UserPageLayout.svelte';
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import { Route } from '$lib/route';
  import { searchNodes } from '$lib/utils/face-graph';
  import { handleError } from '$lib/utils/handle-error';
  import type { FaceGraphNodeDto } from '@immich/sdk';
  import { Button, Checkbox, IconButton, Input, Label, LoadingSpinner, Text } from '@immich/ui';
  import { mdiFitToScreenOutline, mdiRefresh } from '@mdi/js';
  import { onMount } from 'svelte';
  import { t } from 'svelte-i18n';
  import type { PageData } from './$types';
  import FaceGraphHoverCard from './FaceGraphHoverCard.svelte';
  import FaceGraphQueue from './FaceGraphQueue.svelte';
  import FaceGraphSelection from './FaceGraphSelection.svelte';

  interface Props {
    data: PageData;
  }

  let { data }: Props = $props();

  let graph = $state<ReturnType<typeof FaceGraphCanvas>>();
  let selectionPanel = $state<ReturnType<typeof FaceGraphSelection>>();
  let showEdges = $state(true);
  let search = $state('');

  const filters = $derived(faceGraphManager.filters);
  const matches = $derived(searchNodes(faceGraphManager.visibleNodes, search));

  const load = async (force = false) => {
    try {
      await faceGraphManager.load({ force });
    } catch (error) {
      handleError(error, $t('errors.unable_to_load_face_graph'));
    }
  };

  onMount(() => load());

  const onOpen = (node: FaceGraphNodeDto) => goto(Route.faceGraphPerson(node));

  const onSearch = (event: SubmitEvent) => {
    event.preventDefault();
    if (matches.length > 0) {
      faceGraphManager.focusOn(matches[0].id);
    }
  };
</script>

<svelte:document
  use:shortcuts={[
    { shortcut: { key: 'n' }, onShortcut: () => faceGraphManager.step(1) },
    { shortcut: { key: 'p' }, onShortcut: () => faceGraphManager.step(-1) },
    {
      shortcut: { key: 'Enter' },
      onShortcut: () => {
        if (faceGraphManager.selected) {
          void onOpen(faceGraphManager.selected);
        }
      },
    },
    { shortcut: { key: 'r' }, onShortcut: () => selectionPanel?.focusName() },
    { shortcut: { key: 'Escape' }, onShortcut: () => faceGraphManager.clearSelection() },
  ]}
/>

<UserPageLayout title={data.meta.title} description={$t('face_graph_description')} scrollbar={false}>
  {#snippet buttons()}
    <div class="flex items-center gap-1">
      <IconButton
        shape="round"
        variant="ghost"
        color="secondary"
        icon={mdiFitToScreenOutline}
        aria-label={$t('face_graph_fit')}
        onclick={() => graph?.resetView()}
      />
      <Button
        size="small"
        variant="ghost"
        color="secondary"
        leadingIcon={mdiRefresh}
        loading={faceGraphManager.isLoading}
        onclick={() => load(true)}
      >
        {$t('refresh')}
      </Button>
    </div>
  {/snippet}

  <div class="flex h-full flex-col gap-2">
    <div class="flex flex-wrap items-center gap-x-6 gap-y-2 px-2">
      <form onsubmit={onSearch} class="flex w-56 items-center gap-2">
        <Input bind:value={search} size="small" placeholder={$t('search_people')} aria-label={$t('search_people')} />
        {#if search}
          <Text size="tiny" color="muted" class="tabular-nums">{matches.length}</Text>
        {/if}
      </form>

      <div class="flex items-center gap-2">
        <Checkbox id="face-graph-unnamed-only" bind:checked={filters.unnamedOnly} />
        <Label for="face-graph-unnamed-only">{$t('face_graph_unnamed_only')}</Label>
      </div>

      <div class="flex items-center gap-2">
        <Checkbox id="face-graph-show-hidden" bind:checked={filters.showHidden} />
        <Label for="face-graph-show-hidden">{$t('show_hidden_people')}</Label>
      </div>

      <div class="flex items-center gap-2">
        <Checkbox id="face-graph-show-links" bind:checked={showEdges} />
        <Label for="face-graph-show-links">{$t('face_graph_show_links')}</Label>
      </div>

      <div class="flex items-center gap-2">
        <input
          id="face-graph-min-photos"
          type="range"
          min="1"
          max={Math.min(faceGraphManager.maxPhotos, 200)}
          bind:value={filters.minPhotos}
          class="w-32 accent-primary"
        />
        <Label for="face-graph-min-photos" class="tabular-nums">
          {$t('face_graph_min_photos', { values: { count: filters.minPhotos } })}
        </Label>
      </div>
    </div>

    <div class="flex min-h-0 flex-1 gap-2">
      <div class="relative min-w-0 flex-1 rounded-2xl border">
        {#if faceGraphManager.isLoading && faceGraphManager.nodes.length === 0}
          <div class="flex size-full items-center justify-center">
            <LoadingSpinner />
          </div>
        {:else if faceGraphManager.visibleNodes.length === 0}
          <p class="flex size-full items-center justify-center text-dark">{$t('no_people_found')}</p>
        {:else}
          <FaceGraphCanvas
            bind:this={graph}
            nodes={faceGraphManager.visibleNodes}
            edges={faceGraphManager.edges}
            positions={faceGraphManager.positions}
            selectedIds={faceGraphManager.selectedIds}
            focus={faceGraphManager.focus}
            {showEdges}
            label={data.meta.title}
            onSelect={(node, options) => faceGraphManager.select(node.id, options)}
            {onOpen}
            onClear={() => faceGraphManager.clearSelection()}
          >
            {#snippet hoverCard(node)}
              <FaceGraphHoverCard {node} />
            {/snippet}
          </FaceGraphCanvas>
        {/if}
      </div>

      <aside class="hidden w-64 shrink-0 flex-col rounded-2xl border text-dark lg:flex">
        <FaceGraphSelection bind:this={selectionPanel} />
        <div class="min-h-0 flex-1">
          <FaceGraphQueue />
        </div>
      </aside>
    </div>
  </div>
</UserPageLayout>
