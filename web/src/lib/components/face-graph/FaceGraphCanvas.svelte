<script lang="ts">
  import { getAssetMediaUrl, getPeopleThumbnailUrl } from '$lib/utils';
  import { getFaceCrop, isLargeFace, isUnnamed, nodeRadius, type FaceGraphPosition } from '$lib/utils/face-graph';
  import { AssetMediaSize, FaceGraphNodeKind, type FaceGraphEdgeDto, type FaceGraphNodeDto } from '@immich/sdk';
  import { select } from 'd3-selection';
  import { zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3-zoom';
  import { untrack, type Snippet } from 'svelte';

  type Props = {
    nodes: FaceGraphNodeDto[];
    edges: FaceGraphEdgeDto[];
    positions: Map<string, FaceGraphPosition>;
    selectedIds: string[];
    /** the node to move to, a new object for every request */
    focus?: { id: string };
    showEdges?: boolean;
    label: string;
    onSelect: (node: FaceGraphNodeDto, options: { additive: boolean }) => void;
    onOpen: (node: FaceGraphNodeDto) => void;
    onClear: () => void;
    onMove: (node: FaceGraphNodeDto, position: FaceGraphPosition) => void;
    hoverCard?: Snippet<[FaceGraphNodeDto]>;
  };

  let {
    nodes,
    edges,
    positions,
    selectedIds,
    focus,
    showEdges = true,
    label,
    onSelect,
    onOpen,
    onClear,
    onMove,
    hoverCard,
  }: Props = $props();

  // below this size on screen a node is drawn as a dot instead of a thumbnail
  const THUMBNAIL_MIN_RADIUS = 9;
  const LABEL_MIN_RADIUS = 16;
  const EDGE_MIN_SCALE = 0.3;
  const FOCUS_RADIUS = 48;
  const FIT_PADDING = 40;
  // moving the pointer less than this while pressing a node is a click, not a drag
  const DRAG_THRESHOLD = 4;

  let canvas = $state<HTMLCanvasElement>();
  let primaryProbe = $state<HTMLElement>();
  let width = $state(0);
  let height = $state(0);
  let hovered = $state<FaceGraphNodeDto>();
  let pointer = $state({ x: 0, y: 0 });

  let transform: ZoomTransform = zoomIdentity;
  let zoomBehavior: ZoomBehavior<HTMLCanvasElement, unknown> | undefined;
  let frame: number | undefined;
  let hasFit = false;
  // eslint-disable-next-line svelte/prefer-svelte-reactivity
  const images = new Map<string, HTMLImageElement>();

  // small nodes are drawn last so they stay reachable
  const sortedNodes = $derived(nodes.toSorted((a, b) => b.assetCount - a.assetCount));
  const visibleIds = $derived(new Set(nodes.map(({ id }) => id)));
  const selection = $derived(new Set(selectedIds));

  const getImage = (node: FaceGraphNodeDto) => {
    const key = `${node.id}-${node.updatedAt}`;
    let image = images.get(key);
    if (!image) {
      image = new Image();
      image.addEventListener('load', scheduleDraw);
      image.src = node.face
        ? getAssetMediaUrl({
            id: node.face.assetId,
            size: isLargeFace(node.face) ? AssetMediaSize.Thumbnail : AssetMediaSize.Preview,
            edited: false,
          })
        : getPeopleThumbnailUrl(node);
      images.set(key, image);
    }
    return image.complete && image.naturalWidth > 0 ? image : undefined;
  };

  // the paths depend on the zoom and pan, so they cannot be reused between draws
  /* eslint-disable unicorn/prefer-path2d */
  const draw = () => {
    frame = undefined;
    const context = canvas?.getContext('2d');
    if (!canvas || !context || !primaryProbe) {
      return;
    }

    const ratio = window.devicePixelRatio || 1;
    const primary = getComputedStyle(primaryProbe).color;
    const text = getComputedStyle(canvas).color;
    const background = getComputedStyle(document.body).backgroundColor;
    const labels: { name: string; x: number; y: number; maxWidth: number }[] = [];
    const { k } = transform;

    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    if (showEdges && k >= EDGE_MIN_SCALE) {
      context.strokeStyle = text;
      for (const { source, target, distance } of edges) {
        const from = positions.get(source);
        const to = positions.get(target);
        if (!from || !to || !visibleIds.has(source) || !visibleIds.has(target)) {
          continue;
        }
        context.globalAlpha = Math.max(0.08, 0.5 - distance);
        context.lineWidth = Math.max(0.5, 3 * (1 - distance * 2));
        context.beginPath();
        context.moveTo(transform.applyX(from.x), transform.applyY(from.y));
        context.lineTo(transform.applyX(to.x), transform.applyY(to.y));
        context.stroke();
      }
    }

    context.textAlign = 'center';
    context.textBaseline = 'top';
    context.font = '12px sans-serif';

    for (const node of sortedNodes) {
      const position = positions.get(node.id);
      if (!position) {
        continue;
      }

      const x = transform.applyX(position.x);
      const y = transform.applyY(position.y);
      const radius = nodeRadius(node.assetCount) * k;
      if (x + radius < 0 || x - radius > width || y + radius < 0 || y - radius > height) {
        continue;
      }

      const unnamed = isUnnamed(node);
      const isSelected = selection.has(node.id);
      context.globalAlpha = node.isHidden ? 0.35 : 1;
      context.setLineDash(node.kind === FaceGraphNodeKind.Unassigned ? [4, 3] : []);
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);

      if (radius < THUMBNAIL_MIN_RADIUS) {
        context.fillStyle = unnamed ? primary : 'rgb(128 128 128)';
        context.fill();
      } else {
        const image = getImage(node);
        if (image) {
          context.save();
          context.clip();
          if (node.face) {
            // unassigned faces have no thumbnail of their own, the face is cut out of the photo
            const crop = getFaceCrop(node.face);
            const scale = image.naturalWidth / node.face.imageWidth;
            context.drawImage(
              image,
              crop.x * scale,
              crop.y * scale,
              crop.size * scale,
              crop.size * scale,
              x - radius,
              y - radius,
              radius * 2,
              radius * 2,
            );
          } else {
            context.drawImage(image, x - radius, y - radius, radius * 2, radius * 2);
          }
          context.restore();
        } else {
          context.fillStyle = 'rgb(128 128 128 / 0.3)';
          context.fill();
        }
        context.strokeStyle = unnamed ? primary : 'rgb(128 128 128)';
        context.lineWidth = unnamed ? 3 : 1.5;
        context.stroke();
      }

      if (isSelected) {
        context.setLineDash([]);
        context.strokeStyle = primary;
        context.lineWidth = 3;
        context.beginPath();
        context.arc(x, y, radius + 4, 0, Math.PI * 2);
        context.stroke();
      }

      if (node.name && radius >= LABEL_MIN_RADIUS) {
        labels.push({ name: node.name, x, y: y + radius + 6, maxWidth: Math.max(radius * 3, 80) });
      }
    }

    // names are drawn last so they are not covered by other people
    context.globalAlpha = 1;
    context.lineJoin = 'round';
    context.lineWidth = 3;
    context.strokeStyle = background;
    context.fillStyle = text;
    for (const { name, x, y, maxWidth } of labels) {
      context.strokeText(name, x, y, maxWidth);
      context.fillText(name, x, y, maxWidth);
    }

    context.globalAlpha = 1;
    context.setLineDash([]);
  };
  /* eslint-enable unicorn/prefer-path2d */

  function scheduleDraw() {
    frame ??= requestAnimationFrame(draw);
  }

  const findNode = (x: number, y: number) => {
    const [graphX, graphY] = transform.invert([x, y]);
    for (let i = sortedNodes.length - 1; i >= 0; i--) {
      const node = sortedNodes[i];
      const position = positions.get(node.id);
      // dots stay clickable when zoomed out
      const radius = Math.max(nodeRadius(node.assetCount), 5 / transform.k);
      if (position && Math.hypot(position.x - graphX, position.y - graphY) <= radius) {
        return node;
      }
    }
  };

  const setTransform = (next: ZoomTransform) => {
    if (canvas && zoomBehavior) {
      select(canvas).call(zoomBehavior.transform, next);
    }
  };

  const fit = () => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const node of nodes) {
      const position = positions.get(node.id);
      if (!position) {
        continue;
      }
      const radius = nodeRadius(node.assetCount);
      minX = Math.min(minX, position.x - radius);
      minY = Math.min(minY, position.y - radius);
      maxX = Math.max(maxX, position.x + radius);
      maxY = Math.max(maxY, position.y + radius);
    }
    if (!Number.isFinite(minX) || width === 0 || height === 0) {
      return false;
    }

    const scale = Math.min(
      1,
      (width - FIT_PADDING * 2) / Math.max(maxX - minX, 1),
      (height - FIT_PADDING * 2) / Math.max(maxY - minY, 1),
    );
    setTransform(
      zoomIdentity
        .translate(width / 2, height / 2)
        .scale(scale)
        .translate(-(minX + maxX) / 2, -(minY + maxY) / 2),
    );
    return true;
  };

  export const resetView = () => fit();

  $effect(() => {
    if (!canvas) {
      return;
    }

    const element = canvas;
    const pointOf = (event: MouseEvent) => {
      const rect = element.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    let pressed: { node: FaceGraphNodeDto; x: number; y: number; offsetX: number; offsetY: number } | undefined;
    let isDragging = false;
    let wasDragged = false;

    const onPointerDown = (event: PointerEvent) => {
      const { x, y } = pointOf(event);
      const node = event.button === 0 ? findNode(x, y) : undefined;
      const position = node && positions.get(node.id);
      if (!node || !position) {
        return;
      }
      const [graphX, graphY] = transform.invert([x, y]);
      pressed = { node, x, y, offsetX: position.x - graphX, offsetY: position.y - graphY };
      isDragging = false;
      element.setPointerCapture(event.pointerId);
    };

    const onPointerMove = (event: PointerEvent) => {
      pointer = pointOf(event);
      if (!pressed) {
        hovered = findNode(pointer.x, pointer.y);
        return;
      }

      isDragging ||= Math.hypot(pointer.x - pressed.x, pointer.y - pressed.y) > DRAG_THRESHOLD;
      if (isDragging) {
        hovered = undefined;
        const [graphX, graphY] = transform.invert([pointer.x, pointer.y]);
        onMove(pressed.node, { x: graphX + pressed.offsetX, y: graphY + pressed.offsetY });
      }
    };

    const onPointerUp = (event: PointerEvent) => {
      if (!pressed) {
      	return;
      }

      element.releasePointerCapture(event.pointerId);
      // the click that follows a drag must not change the selection
      wasDragged = isDragging;
      pressed = undefined;
      isDragging = false;
    };
    const onPointerLeave = () => (hovered = undefined);
    const onClick = (event: MouseEvent) => {
      if (wasDragged) {
        wasDragged = false;
        return;
      }
      const { x, y } = pointOf(event);
      const node = findNode(x, y);
      if (node) {
        onSelect(node, { additive: event.shiftKey });
      } else {
        onClear();
      }
    };
    const onDoubleClick = (event: MouseEvent) => {
      const { x, y } = pointOf(event);
      const node = findNode(x, y);
      if (node) {
        onOpen(node);
      }
    };

    zoomBehavior = zoom<HTMLCanvasElement, unknown>()
      // pressing a node drags the node, pressing the background pans the graph
      .filter((event: MouseEvent | WheelEvent | TouchEvent) => {
        if (event.type === 'wheel') {
          return true;
        }
        const point = 'touches' in event ? event.touches[0] : (event as MouseEvent);
        const rect = element.getBoundingClientRect();
        const isOnNode = !!point && !!findNode(point.clientX - rect.left, point.clientY - rect.top);
        return !isOnNode && !('button' in event && event.button);
      })
      .scaleExtent([0.02, 8])
      .on('zoom', (event: { transform: ZoomTransform }) => {
        transform = event.transform;
        hovered = undefined;
        scheduleDraw();
      });
    select(element).call(zoomBehavior).on('dblclick.zoom', null);

    element.addEventListener('pointerdown', onPointerDown);
    element.addEventListener('pointerup', onPointerUp);
    element.addEventListener('pointercancel', onPointerUp);
    element.addEventListener('pointermove', onPointerMove);
    element.addEventListener('pointerleave', onPointerLeave);
    element.addEventListener('click', onClick);
    element.addEventListener('dblclick', onDoubleClick);

    return () => {
      select(element).on('.zoom', null);
      element.removeEventListener('pointerdown', onPointerDown);
      element.removeEventListener('pointerup', onPointerUp);
      element.removeEventListener('pointercancel', onPointerUp);
      element.removeEventListener('pointermove', onPointerMove);
      element.removeEventListener('pointerleave', onPointerLeave);
      element.removeEventListener('click', onClick);
      element.removeEventListener('dblclick', onDoubleClick);
      if (frame !== undefined) {
        cancelAnimationFrame(frame);
        frame = undefined;
      }
    };
  });

  $effect(() => {
    if (!canvas || width === 0 || height === 0) {
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    if (!hasFit && positions.size > 0) {
      hasFit = fit();
    }
    scheduleDraw();
  });

  $effect(() => {
    // redraw whenever something that is drawn changes
    void [sortedNodes, edges, positions, selection, showEdges];
    scheduleDraw();
  });

  $effect(() => {
    const id = focus?.id;
    untrack(() => {
      const position = id && positions.get(id);
      const node = nodes.find((node) => node.id === id);
      if (!position || !node) {
        return;
      }
      const scale = Math.min(8, Math.max(transform.k, FOCUS_RADIUS / nodeRadius(node.assetCount)));
      setTransform(
        zoomIdentity
          .translate(width / 2, height / 2)
          .scale(scale)
          .translate(-position.x, -position.y),
      );
    });
  });
</script>

<div class="relative size-full overflow-hidden" bind:clientWidth={width} bind:clientHeight={height}>
  <span bind:this={primaryProbe} class="hidden text-primary"></span>
  <canvas
    bind:this={canvas}
    aria-label={label}
    class="absolute inset-0 size-full touch-none text-dark {hovered ? 'cursor-pointer' : 'cursor-grab'}"
  ></canvas>

  {#if hovered && hoverCard}
    <div
      class="pointer-events-none absolute z-10"
      style:left="{Math.min(pointer.x + 16, Math.max(width - 260, 0))}px"
      style:top="{Math.min(pointer.y + 16, Math.max(height - 220, 0))}px"
    >
      {@render hoverCard(hovered)}
    </div>
  {/if}
</div>
