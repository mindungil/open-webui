<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { getUsageSummary } from '$lib/apis/usage';
	export let className = '';
		interface UsageData {
		name: string;
		percent: string;
		dailyTokensPercent: string;
		dailyRequestsPercent: string;
		monthlyTokensPercent: string;
	}

	interface ApiUsage {
		api_type: string;
		daily: {
			tokens_percent: number;
			requests_percent: number;
		};
		monthly: {
			tokens_percent: number;
		};
	}

	let usageData: UsageData[] = [];
	let interval: ReturnType<typeof setInterval> | null = null;
	let expanded = false;
	let dropUp = false;

	const loadUsage = async () => {
		try {
			const token = localStorage.getItem('token');
			if (!token) {
				console.log('No token found');
				usageData = [];
				return;
			}
			const response = await getUsageSummary(token);
			if (response?.success && response.data?.has_limits) {
				usageData = response.data.api_usage.map((usage: ApiUsage) => ({
					name: usage.api_type.toUpperCase(),
					percent: usage.monthly.tokens_percent.toFixed(0),
					dailyTokensPercent: usage.daily.tokens_percent.toFixed(0),
					dailyRequestsPercent: usage.daily.requests_percent.toFixed(0),
					monthlyTokensPercent: usage.monthly.tokens_percent.toFixed(0),
				}));
			} else {
				usageData = [];
			}
		} catch (error) {
			console.log('Usage data load failed:', error);
			usageData = [];
		}
	};

	onMount(() => {
		loadUsage();
		interval = setInterval(loadUsage, 5000); // 30초마다 업데이트
	});

	onDestroy(() => {
		if (interval) clearInterval(interval);
	});

	function handleExpand(e: MouseEvent) {
		expanded = true;
		// 안내 박스가 화면 하단에 가까우면 위로 뜨게
		const target = e.currentTarget as HTMLElement | null;
		if (target && typeof target.getBoundingClientRect === 'function') {
			const rect = target.getBoundingClientRect();
			const spaceBelow = window.innerHeight - rect.bottom;
			dropUp = spaceBelow < 180; // 안내 박스 높이보다 공간이 부족하면 위로
		} else {
			dropUp = false;
		}
	}

	$: total = usageData.reduce(
		(acc, cur) => ({
			dailyTokensPercent: acc.dailyTokensPercent + Number(cur.dailyTokensPercent),
			dailyRequestsPercent: acc.dailyRequestsPercent + Number(cur.dailyRequestsPercent),
			monthlyTokensPercent: acc.monthlyTokensPercent + Number(cur.monthlyTokensPercent),
		}),
		{ dailyTokensPercent: 0, dailyRequestsPercent: 0, monthlyTokensPercent: 0 }
	);
</script>

{#if usageData.length > 0}
	<div>
		<div
			class={`flex flex-row items-center gap-1 px-2 py-0.5 min-w-fit rounded-full bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 cursor-pointer shadow-sm text-xs font-medium w-auto ${className}`}
			on:mouseenter={handleExpand}
			on:mouseleave={() => expanded = false}
			style="height: 28px; min-height: 0; max-height: 32px;"
		>
			<div class="w-2 h-2 rounded-full shrink-0 mr-1 {+total.dailyTokensPercent >= 90 ? 'bg-red-500' : +total.dailyTokensPercent >= 70 ? 'bg-yellow-500' : 'bg-green-500'}"></div>
			<span class="text-gray-600 dark:text-gray-400">API 사용량: {total.dailyTokensPercent}%</span>
		</div>

		{#if expanded}
			<div class="absolute right-0 {dropUp ? 'bottom-full mb-2' : 'mt-2'} w-64 border rounded shadow-lg p-3 bg-white dark:bg-gray-800 animate-fade-in">
				<div class="mb-2">
					<div class="font-bold mb-1">외부 API 사용량</div>
					<div class="flex flex-col gap-0.5 text-xs">
						<div class="flex items-center gap-1">
							<div class="w-2 h-2 rounded-full {+total.dailyTokensPercent >= 90 ? 'bg-red-500' : +total.dailyTokensPercent >= 70 ? 'bg-yellow-500' : 'bg-green-500'}"></div>
							<span>일일 토큰 사용량:</span> <span class="font-mono">{total.dailyTokensPercent}%</span>
						</div>
						<div class="flex items-center gap-1">
							<div class="w-2 h-2 rounded-full {+total.dailyRequestsPercent >= 90 ? 'bg-red-500' : +total.dailyRequestsPercent >= 70 ? 'bg-yellow-500' : 'bg-green-500'}"></div>
							<span>일일 요청 사용량:</span> <span class="font-mono">{total.dailyRequestsPercent}%</span>
						</div>
						<div class="flex items-center gap-1">
							<div class="w-2 h-2 rounded-full {+total.monthlyTokensPercent >= 90 ? 'bg-red-500' : +total.monthlyTokensPercent >= 70 ? 'bg-yellow-500' : 'bg-green-500'}"></div>
							<span>월간 토큰 사용량:</span> <span class="font-mono">{total.monthlyTokensPercent}%</span>
						</div>
					</div>
				</div>
			</div>
		{/if}
	</div>
{/if}
