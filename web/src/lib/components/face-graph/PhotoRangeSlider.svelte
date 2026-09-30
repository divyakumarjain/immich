<script lang="ts">
  import { photosToSlider, SLIDER_STEPS, sliderToPhotos } from '$lib/utils/face-graph';

  type Props = {
    /** the largest number of photos any person has */
    limit: number;
    min: number;
    /** `undefined` means no upper limit */
    max: number | undefined;
    minLabel: string;
    maxLabel: string;
  };

  let { limit, min = $bindable(), max = $bindable(), minLabel, maxLabel }: Props = $props();

  const low = $derived(photosToSlider(min, limit));
  const high = $derived(max === undefined ? SLIDER_STEPS : photosToSlider(max, limit));

  const onLow = (event: Event & { currentTarget: HTMLInputElement }) => {
    const position = Math.min(Number(event.currentTarget.value), high);
    event.currentTarget.value = String(position);
    min = sliderToPhotos(position, limit);
  };

  const onHigh = (event: Event & { currentTarget: HTMLInputElement }) => {
    const position = Math.max(Number(event.currentTarget.value), low);
    event.currentTarget.value = String(position);
    max = position >= SLIDER_STEPS ? undefined : sliderToPhotos(position, limit);
  };

  // the two inputs lie on top of each other, only their handles react to the pointer
  const inputClass =
    'pointer-events-none absolute inset-0 m-0 h-full w-full cursor-pointer appearance-none bg-transparent ' +
    '[&::-webkit-slider-runnable-track]:appearance-none [&::-webkit-slider-runnable-track]:bg-transparent ' +
    '[&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:size-4 ' +
    '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full ' +
    '[&::-webkit-slider-thumb]:bg-primary [&::-moz-range-thumb]:pointer-events-auto ' +
    '[&::-moz-range-thumb]:size-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 ' +
    '[&::-moz-range-thumb]:bg-primary [&::-moz-range-track]:bg-transparent';
</script>

<div class="relative h-5 w-44">
  <div class="absolute inset-x-2 top-1/2 h-1 -translate-y-1/2 rounded-full bg-gray-300 dark:bg-gray-600">
    <div
      class="absolute h-full rounded-full bg-primary"
      style:left="{(low / SLIDER_STEPS) * 100}%"
      style:right="{100 - (high / SLIDER_STEPS) * 100}%"
    ></div>
  </div>
  <input
    type="range"
    min="0"
    max={SLIDER_STEPS}
    value={low}
    oninput={onLow}
    aria-label={minLabel}
    class="{inputClass} {low > SLIDER_STEPS / 2 ? 'z-10' : ''}"
  />
  <input
    type="range"
    min="0"
    max={SLIDER_STEPS}
    value={high}
    oninput={onHigh}
    aria-label={maxLabel}
    class={inputClass}
  />
</div>
