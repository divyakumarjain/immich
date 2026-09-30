<script lang="ts">
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import { getPeopleThumbnailUrl } from '$lib/utils';
  import type { FaceGraphNodeDto } from '@immich/sdk';
  import { Text } from '@immich/ui';
  import { t } from 'svelte-i18n';

  type Props = {
    node: FaceGraphNodeDto;
  };

  let { node }: Props = $props();

  const neighbors = $derived(faceGraphManager.getNeighbors(node.id, 3));
</script>

<div class="w-60 rounded-2xl border bg-light p-3 text-dark shadow-lg">
  <div class="flex items-center gap-3">
    <img src={getPeopleThumbnailUrl(node)} alt={node.name} class="size-14 rounded-full object-cover" />
    <div class="min-w-0">
      <p class="truncate font-medium">{node.name || $t('add_a_name')}</p>
      <Text size="tiny" color="muted">
        {$t('face_graph_photos_and_faces', { values: { photos: node.assetCount, faces: node.faceCount } })}
      </Text>
    </div>
  </div>

  {#if neighbors.length > 0}
    <Text size="tiny" color="muted" fontWeight="medium" class="mt-3 mb-1">{$t('face_graph_similar_people')}</Text>
    {#each neighbors as neighbor (neighbor.node.id)}
      <div class="flex items-center gap-2 py-0.5 text-sm">
        <img
          src={getPeopleThumbnailUrl(neighbor.node)}
          alt={neighbor.node.name}
          class="size-6 rounded-full object-cover"
        />
        <span class="min-w-0 flex-1 truncate">{neighbor.node.name || $t('add_a_name')}</span>
        <span class="text-xs tabular-nums opacity-70">{neighbor.distance.toFixed(2)}</span>
      </div>
    {/each}
  {/if}
</div>
