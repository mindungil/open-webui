<script lang="ts">
	import { goto } from '$app/navigation';
	import { user } from '$lib/stores';
	import { onMount } from 'svelte';

	onMount(() => {
		if ($user?.role !== 'admin') {
			if ($user?.permissions?.workspace?.models) {
				goto('/gpts/models');
			} else if ($user?.permissions?.workspace?.knowledge) {
				goto('/gpts/knowledge');
			} else {
				goto('/');
			}
		} else {
			goto('/gpts/models');
		}
	});
</script>
