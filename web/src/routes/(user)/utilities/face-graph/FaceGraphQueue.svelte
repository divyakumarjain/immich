<script lang="ts">
  import { faceGraphManager } from '$lib/managers/face-graph-manager.svelte';
  import FaceGraphNodeThumbnail from '$lib/components/face-graph/FaceGraphNodeThumbnail.svelte';
  import { isUnassigned } from '$lib/utils/face-graph';
  import { Text } from '@immich/ui';
  import { t } from 'svelte-i18n';

  // the queue can hold thousands of people, the rest is reachable with the keyboard
  const MAX_ITEMS = 100;

  const queue = $derived(faceGraphManager.queue);
  const selectedId = $derived(faceGraphManager.selected?.id);
</script>

<section class="flex h-full flex-col">
  <div class="p-3">
    <Text fontWeight="medium">{$t('face_graph_unnamed_queue')}</Text>
    <Text size="tiny" color="muted">{$t('face_graph_unnamed_left', { values: { count: queue.length } })}</Text>
  </div>

  <ul class="min-h-0 flex-1 immich-scrollbar overflow-y-auto">
    {#each queue.slice(0, MAX_ITEMS) as node (node.id)}
      <li>
        <button
          type="button"
          class="flex w-full items-center gap-3 px-3 py-1.5 text-start hover:bg-gray-100 dark:hover:bg-immich-dark-gray
            {node.id === selectedId ? 'bg-primary/10' : ''}"
          onclick={() => faceGraphManager.focusOn(node.id)}
        >
          <FaceGraphNodeThumbnail {node} size={40} />
          <span class="text-sm">
            {$t('face_graph_photos', { values: { count: node.assetCount } })}
            {#if isUnassigned(node)}
              <span class="block text-xs opacity-70">{$t('face_unassigned')}</span>
            {/if}
          </span>
        </button>
      </li>
    {/each}
  </ul>
</section>
