<script lang="ts">
  import { getPeopleThumbnailUrl } from '$lib/utils';
  import { handleError } from '$lib/utils/handle-error';
  import { normalizeSearchString } from '$lib/utils/string-utils';
  import { searchPerson, type PersonResponseDto } from '@immich/sdk';
  import { Button, Input } from '@immich/ui';
  import { t } from 'svelte-i18n';

  type Props = {
    /** the person that is being named, left out of the suggestions */
    personId: string;
    name: string;
    disabled?: boolean;
    ref?: HTMLInputElement | null;
    onSave: (name: string) => Promise<void> | void;
    /** an existing person was chosen: merge into it, returns whether that happened */
    onPick: (person: PersonResponseDto) => Promise<boolean>;
  };

  let { personId, name, disabled = false, ref = $bindable(null), onSave, onPick }: Props = $props();

  const MAX_SUGGESTIONS = 5;
  const SEARCH_DELAY_MS = 200;

  let value = $derived(name);
  let suggestions = $state<PersonResponseDto[]>([]);
  let isOpen = $state(false);
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let abortController: AbortController | undefined;

  const cancelSearch = () => {
    clearTimeout(timeout);
    abortController?.abort();
    abortController = undefined;
  };

  const search = async (query: string) => {
    abortController = new AbortController();
    try {
      const people = await searchPerson({ name: query }, { signal: abortController.signal });
      suggestions = people.filter((person) => person.id !== personId && person.name).slice(0, MAX_SUGGESTIONS);
    } catch (error) {
      handleError(error, $t('errors.cant_search_people'));
    }
  };

  const onInput = () => {
    cancelSearch();
    isOpen = true;
    const query = value.trim();
    if (!query) {
      suggestions = [];
      return;
    }
    timeout = setTimeout(() => void search(query), SEARCH_DELAY_MS);
  };

  const close = () => {
    cancelSearch();
    isOpen = false;
    suggestions = [];
  };

  const pick = async (person: PersonResponseDto) => {
    close();
    return onPick(person);
  };

  const onsubmit = async (event: SubmitEvent) => {
    event.preventDefault();
    const newName = value.trim();
    if (newName === name) {
      return;
    }

    // typing the name of someone who already exists most likely means they are the same person
    const existing = suggestions.find(
      (person) => normalizeSearchString(person.name.trim()) === normalizeSearchString(newName),
    );
    close();
    if (existing && (await onPick(existing))) {
      return;
    }
    await onSave(newName);
  };

  const onkeydown = (event: KeyboardEvent) => {
    if (!(event.key === 'Escape' && isOpen)) {
      return;
    }

    event.stopPropagation();
    close();
  };
</script>

<form {onsubmit} class="relative flex items-center gap-2" autocomplete="off">
  <Input
    bind:ref
    bind:value
    size="small"
    placeholder={$t('add_a_name')}
    aria-label={$t('name')}
    {disabled}
    oninput={onInput}
    {onkeydown}
    onblur={() => setTimeout(() => (isOpen = false), 150)}
  />
  <Button type="submit" size="small" disabled={disabled || value.trim() === name}>{$t('save')}</Button>

  {#if isOpen && suggestions.length > 0}
    <ul class="absolute inset-x-0 top-full z-20 mt-1 overflow-hidden rounded-xl border bg-light text-dark shadow-lg">
      {#each suggestions as person (person.id)}
        <li>
          <button
            type="button"
            class="flex w-full items-center gap-3 px-3 py-2 text-start hover:bg-gray-100 dark:hover:bg-immich-dark-gray"
            onmousedown={(event) => event.preventDefault()}
            onclick={() => pick(person)}
          >
            <img src={getPeopleThumbnailUrl(person)} alt="" class="size-8 rounded-full object-cover" />
            <span class="min-w-0 flex-1 truncate text-sm">{person.name}</span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</form>
