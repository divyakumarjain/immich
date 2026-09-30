import { getFaceGraph, type FaceGraphEdgeDto, type FaceGraphNodeDto } from '@immich/sdk';
import {
  createLayout,
  getNeighbors,
  getWorkQueue,
  matchesFilters,
  type FaceGraphFilters,
  type FaceGraphLayout,
  type FaceGraphPosition,
} from '$lib/utils/face-graph';

// rearranging runs several steps per frame so it settles in about a second
const REARRANGE_TICKS_PER_FRAME = 4;

class FaceGraphManager {
  #nodes = $state.raw<FaceGraphNodeDto[]>([]);
  #edges = $state.raw<FaceGraphEdgeDto[]>([]);
  #positions = $state.raw(new Map<string, FaceGraphPosition>());
  #selectedIds = $state.raw<string[]>([]);
  #focus = $state.raw<{ id: string }>();
  #layout: FaceGraphLayout | undefined;
  #frame: number | undefined;
  #isRearranging = false;
  #rearranged = $state.raw<object>();
  #isLoaded = false;
  #hasUnassigned = false;

  isLoading = $state(false);
  filters = $state<FaceGraphFilters>({ unnamedOnly: false, showHidden: false, showUnassigned: false, minPhotos: 1 });

  readonly nodeById = $derived(new Map(this.#nodes.map((node) => [node.id, node])));
  readonly visibleNodes = $derived(this.#nodes.filter((node) => matchesFilters(node, this.filters)));
  readonly queue = $derived(getWorkQueue(this.visibleNodes));
  readonly selected = $derived(this.nodeById.get(this.#selectedIds.at(-1) ?? ''));
  readonly maxPhotos = $derived(Math.max(1, ...this.#nodes.map(({ assetCount }) => assetCount)));

  get nodes() {
    return this.#nodes;
  }

  get edges() {
    return this.#edges;
  }

  get positions() {
    return this.#positions;
  }

  get selectedIds() {
    return this.#selectedIds;
  }

  /** the node the graph should move to, a new object for every request */
  get focus() {
    return this.#focus;
  }

  /** changes to a new object every time the graph has settled after it was rearranged */
  get rearranged() {
    return this.#rearranged;
  }

  async load({ force = false }: { force?: boolean } = {}) {
    const withUnassigned = this.filters.showUnassigned;
    // grouping the unassigned faces is slow, so they are only loaded when they are shown
    if (this.#isLoaded && !force && (this.#hasUnassigned || !withUnassigned)) {
      return;
    }

    this.isLoading = true;
    try {
      const { nodes, edges } = await getFaceGraph({ withHidden: true, withUnassigned });
      if (this.#frame !== undefined) {
        cancelAnimationFrame(this.#frame);
        this.#frame = undefined;
      }
      this.#nodes = nodes;
      this.#edges = edges;
      this.#layout = createLayout(nodes, edges, this.visibleNodes);
      this.#positions = this.#layout.getPositions();
      this.#selectedIds = this.#selectedIds.filter((id) => this.nodeById.has(id));
      this.#isLoaded = true;
      this.#hasUnassigned = withUnassigned;
    } finally {
      this.isLoading = false;
    }
  }

  /** the next load fetches the graph again, e.g. after faces were moved between people */
  invalidate() {
    this.#isLoaded = false;
  }

  /** moves a node, it stays where it is dropped and the other nodes make room */
  moveNode(id: string, position: FaceGraphPosition) {
    if (!this.#layout) {
      return;
    }
    this.#layout.pin(id, position);
    this.#frame ??= requestAnimationFrame(() => this.#animate());
  }

  #animate() {
    this.#frame = undefined;
    if (!this.#layout) {
      return;
    }
    const isMoving = this.#layout.step(this.#isRearranging ? REARRANGE_TICKS_PER_FRAME : 1);
    this.#positions = this.#layout.getPositions();
    if (isMoving) {
      this.#frame = requestAnimationFrame(() => this.#animate());
    } else if (this.#isRearranging) {
      this.#isRearranging = false;
      this.#rearranged = {};
    }
  }

  /**
   * Lays the graph out again for the people that are shown, so filtered people leave no gaps.
   * Call it whenever `visibleNodes` changes.
   */
  rearrange() {
    if (!this.#layout?.show(this.visibleNodes)) {
      return;
    }
    this.#isRearranging = true;
    this.#positions = this.#layout.getPositions();
    this.#frame ??= requestAnimationFrame(() => this.#animate());
  }

  select(id: string, { additive = false }: { additive?: boolean } = {}) {
    if (!additive) {
      this.#selectedIds = [id];
      return;
    }
    this.#selectedIds = this.#selectedIds.includes(id)
      ? this.#selectedIds.filter((selectedId) => selectedId !== id)
      : [...this.#selectedIds, id];
  }

  clearSelection() {
    this.#selectedIds = [];
  }

  focusOn(id: string) {
    this.select(id);
    this.#focus = { id };
  }

  /** moves through the unnamed people, the ones with the most photos first */
  step(offset: 1 | -1) {
    if (this.queue.length === 0) {
      return;
    }
    const index = this.queue.findIndex(({ id }) => id === this.selected?.id);
    const next = index === -1 ? (offset === 1 ? 0 : this.queue.length - 1) : index + offset;
    const node = this.queue[Math.min(this.queue.length - 1, Math.max(0, next))];
    this.focusOn(node.id);
  }

  getNeighbors(id: string, limit: number) {
    return getNeighbors(id, this.#edges)
      .map(({ id, distance }) => ({ node: this.nodeById.get(id), distance }))
      .filter((neighbor): neighbor is { node: FaceGraphNodeDto; distance: number } => !!neighbor.node)
      .slice(0, limit);
  }

  updateNode(id: string, changes: Partial<FaceGraphNodeDto>) {
    this.#nodes = this.#nodes.map((node) => (node.id === id ? { ...node, ...changes } : node));
  }

  /** adds the photos of the other people to the target and removes them from the graph */
  mergeNodes(targetId: string, otherIds: string[]) {
    const others = otherIds.map((id) => this.nodeById.get(id)).filter((node) => !!node);
    const target = this.nodeById.get(targetId);
    if (target) {
      this.updateNode(targetId, {
        assetCount: target.assetCount + others.reduce((sum, { assetCount }) => sum + assetCount, 0),
        faceCount: target.faceCount + others.reduce((sum, { faceCount }) => sum + faceCount, 0),
      });
    }
    this.removeNodes(otherIds);
    this.#selectedIds = [targetId];
  }

  removeNodes(ids: string[]) {
    this.#nodes = this.#nodes.filter((node) => !ids.includes(node.id));
    this.#edges = this.#edges.filter(({ source, target }) => !ids.includes(source) && !ids.includes(target));
    this.#selectedIds = this.#selectedIds.filter((id) => !ids.includes(id));
  }
}

export const faceGraphManager = new FaceGraphManager();
