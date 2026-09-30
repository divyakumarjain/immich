<script lang="ts">
  import { getAssetMediaUrl } from '$lib/utils';
  import { getFaceCropStyle, isLargeFace } from '$lib/utils/face-graph';
  import { AssetMediaSize, type FaceGroupFaceDto } from '@immich/sdk';

  type Props = {
    face: FaceGroupFaceDto;
    size: number;
  };

  let { face, size }: Props = $props();

  const style = $derived(getFaceCropStyle(face, size));
  // the bounding box refers to the image without edits
  const url = $derived(
    getAssetMediaUrl({
      id: face.assetId,
      size: isLargeFace(face) ? AssetMediaSize.Thumbnail : AssetMediaSize.Preview,
      edited: false,
    }),
  );
</script>

<div class="relative overflow-hidden bg-gray-200 dark:bg-gray-700" style:width="{size}px" style:height="{size}px">
  <img
    src={url}
    alt=""
    loading="lazy"
    draggable="false"
    class="absolute max-w-none"
    style:width="{style.width}px"
    style:height="{style.height}px"
    style:left="{style.left}px"
    style:top="{style.top}px"
  />
</div>
