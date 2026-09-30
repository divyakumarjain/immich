<script lang="ts">
  import FaceCrop from '$lib/components/face-graph/FaceCrop.svelte';
  import { getPeopleThumbnailUrl } from '$lib/utils';
  import type { FaceGraphNodeDto } from '@immich/sdk';

  type Props = {
    node: FaceGraphNodeDto;
    /** size in pixels */
    size: number;
  };

  let { node, size }: Props = $props();
</script>

<div
  class="shrink-0 overflow-hidden rounded-full {node.isHidden ? 'opacity-40' : ''}"
  style:width="{size}px"
  style:height="{size}px"
>
  {#if node.face}
    <FaceCrop face={node.face} {size} />
  {:else}
    <img src={getPeopleThumbnailUrl(node)} alt={node.name} loading="lazy" class="size-full object-cover" />
  {/if}
</div>
