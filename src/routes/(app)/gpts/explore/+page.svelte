<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { getGPTTemplates, addGPTTemplateToUser, removeGPTTemplateFromUser, getUserGPTTemplates } from '$lib/apis/gpt-templates';

	import { user, userGPTs } from '$lib/stores';
	import { goto } from '$app/navigation';

	const i18n = getContext('i18n');

	let loaded = false;
	let templates = [];
	let query = '';

	const init = async () => {
		try {
			const allTemplates = await getGPTTemplates(localStorage.token);
			const myTemplates = await getUserGPTTemplates(localStorage.token);
			const myTemplateIds = new Set((myTemplates || []).map(t => t.id));

			templates = (allTemplates || []).map(template => ({
				...template,
				tags: template.tags || [],
				isActive: myTemplateIds.has(template.id)
			}));

			userGPTs.set(myTemplates || []);
		} catch (error) {
			console.error('Failed to load templates:', error);
			templates = [];
		}
	};
	const toggleTemplate = async (template) => {
		try {
			if (template.isActive) {
				await removeGPTTemplateFromUser(localStorage.token, template.id);
				template.isActive = false;
			} else {
				await addGPTTemplateToUser(localStorage.token, template.id);
				template.isActive = true;
			}
			const myTemplates = await getUserGPTTemplates(localStorage.token);
			userGPTs.set(myTemplates || []);
			templates = templates.map(t => 
				t.id === template.id ? { ...t, isActive: template.isActive } : t
			);
		} catch (error) {
			console.error('Failed to toggle template:', error);
		}
	};


	onMount(async () => {
		await init();
		loaded = true;
	});

	// 검색 필터
	$: filteredTemplates = templates.filter((template) => {
		if (!query) return true;
		const searchLower = query.toLowerCase();
		return (
			template.name.toLowerCase().includes(searchLower) ||
			template.description.toLowerCase().includes(searchLower) ||
			template.tags.some((tag) => tag.toLowerCase().includes(searchLower))
		);
	});
</script>

{#if loaded}
	<div class="flex flex-col h-full">
		<!-- 헤더 섹션 -->
		<div class="mb-6 pt-4">
			<div class="flex flex-col gap-2">
				<div class="text-2xl font-semibold">{$i18n.t('Explore GPT Templates')}</div>
				<div class="text-sm text-gray-500 dark:text-gray-400">
					{$i18n.t('Discover and use pre-built GPT templates')}
				</div>
			</div>
		</div>

		<!-- 검색 바 -->
		<div class="mb-6">
			<div class="flex gap-2">
				<input
					type="text"
					bind:value={query}
					placeholder={$i18n.t('Search templates...')}
					class="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
				/>
			</div>
		</div>

		<!-- 템플릿 그리드 -->
		<div class="flex-1 overflow-y-auto">
			{#if filteredTemplates.length > 0}
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
					{#each filteredTemplates as template}
						<a
							href="/gpts/explore/{template.id}"
							class="group block p-5 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 bg-white dark:bg-gray-800 hover:shadow-lg transition-all duration-200"
						>
							<!-- 아이콘과 이름 -->
							<div class="flex items-start gap-3 mb-3">
								<div class="text-4xl">{template.icon}</div>
								<div class="flex-1 min-w-0">
									<div
										class="font-semibold text-lg group-hover:text-blue-600 dark:group-hover:text-blue-400 transition truncate"
									>
										{template.name}
									</div>
									<div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
										{template.usage_count.toLocaleString()} {$i18n.t('uses')}
									</div>
								</div>
							</div>

							<!-- 설명 -->
							<div class="text-sm text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
								{template.description}
							</div>

							<!-- 태그 -->
							<div class="flex flex-wrap gap-1.5">
								{#each template.tags as tag}
									<span
										class="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
									>
										{tag}
									</span>
								{/each}
							</div>
						</a>
					{/each}
				</div>
			{:else}
				<div class="text-center py-12">
					<div class="text-gray-500 dark:text-gray-400">
						{$i18n.t('No templates found')}
					</div>
				</div>
			{/if}
		</div>
	</div>
{:else}
	<!-- 로딩 상태 -->
	<div class="flex items-center justify-center h-full">
		<div class="text-gray-500 dark:text-gray-400">{$i18n.t('Loading...')}</div>
	</div>
{/if}
