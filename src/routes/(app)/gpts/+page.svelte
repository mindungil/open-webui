<script lang="ts">
	import { goto } from '$app/navigation';
	import { user } from '$lib/stores';
	import { onMount } from 'svelte';

	onMount(() => {
		// 기본적으로 탐색하기 페이지로 리다이렉트
		goto('/gpts/explore');
	});
</script>
