<script lang="ts">
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import FaceGraphNodeThumbnail from '$lib/components/face-graph/FaceGraphNodeThumbnail.svelte';
  import { isUnassigned } from '$lib/utils/face-graph';
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
    <FaceGraphNodeThumbnail {node} size={56} />
    <div class="min-w-0">
      <p class="truncate font-medium">
        {isUnassigned(node) ? $t('face_graph_unassigned_faces') : node.name || $t('add_a_name')}
      </p>
      <Text size="tiny" color="muted">
        {$t('face_graph_photos_and_faces', { values: { photos: node.assetCount, faces: node.faceCount } })}
      </Text>
    </div>
  </div>

  {#if neighbors.length > 0}
    <Text size="tiny" color="muted" fontWeight="medium" class="mt-3 mb-1">{$t('face_graph_similar_people')}</Text>
    {#each neighbors as neighbor (neighbor.node.id)}
      <div class="flex items-center gap-2 py-0.5 text-sm">
        <FaceGraphNodeThumbnail node={neighbor.node} size={24} />
        <span class="min-w-0 flex-1 truncate">
          {isUnassigned(neighbor.node) ? $t('face_graph_unassigned_faces') : neighbor.node.name || $t('add_a_name')}
        </span>
        <span class="text-xs tabular-nums opacity-70">{neighbor.distance.toFixed(2)}</span>
      </div>
    {/each}
  {/if}
</div>
