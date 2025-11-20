<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { user, userGPTs } from '$lib/stores';
	import { getGPTTemplateById, addGPTTemplateToUser, removeGPTTemplateFromUser, getUserGPTTemplates } from '$lib/apis/gpt-templates';
	import { toast } from 'svelte-sonner';

	const i18n = getContext('i18n');

	let loaded = false;
	let template = null;
	let isAdded = false;

	const init = async () => {
		const templateId = $page.params.id;

		try {
			template = await getGPTTemplateById(localStorage.token, templateId);
			if (!template) {
				goto('/gpts/explore');
				return;
			}

			// 이미 추가된 템플릿인지 확인
			const myTemplates = await getUserGPTTemplates(localStorage.token);
			userGPTs.set(myTemplates || []);
			isAdded = (myTemplates || []).some(t => t.id === template.id);
		} catch (error) {
			console.error('Failed to load template:', error);
			goto('/gpts/explore');
		}
	};

	const handleToggle = async () => {
		try {
			if (isAdded) {
				await removeGPTTemplateFromUser(localStorage.token, template.id);
				isAdded = false;
				toast.success(`"${template.name}" 템플릿이 삭제되었습니다.`);
			} else {
				await addGPTTemplateToUser(localStorage.token, template.id);
				isAdded = true;
				toast.success(`"${template.name}" 템플릿이 추가되었습니다.`);
			}
			const myTemplates = await getUserGPTTemplates(localStorage.token);
			userGPTs.set(myTemplates || []);
		} catch (error) {
			console.error('Failed to toggle template:', error);
			toast.error(isAdded ? '템플릿 삭제에 실패했습니다.' : '템플릿 추가에 실패했습니다.');
		}
	};

	onMount(async () => {
		await init();
		loaded = true;
	});
</script>

{#if loaded && template}
	<div class="max-w-4xl mx-auto py-8">
		<!-- 뒤로 가기 버튼 -->
		<button
			on:click={() => goto('/gpts/explore')}
			class="mb-6 flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition"
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				class="h-5 w-5"
				viewBox="0 0 20 20"
				fill="currentColor"
			>
				<path
					fill-rule="evenodd"
					d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z"
					clip-rule="evenodd"
				/>
			</svg>
			<span>{$i18n.t('Back to Templates')}</span>
		</button>

		<!-- 템플릿 헤더 -->
		<div class="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-8 mb-6">
			<div class="flex items-start gap-6">
				<div class="text-6xl">{template.icon}</div>
				<div class="flex-1">
					<h1 class="text-3xl font-bold mb-2">{template.name}</h1>
					<p class="text-gray-600 dark:text-gray-400 mb-4">{template.description}</p>

					<div class="flex items-center gap-4 mb-4">
						<!-- 평점 -->
						{#if template.rating_avg}
							<div class="flex items-center gap-1">
								<svg class="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
									<path
										d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"
									/>
								</svg>
								<span class="font-semibold">{template.rating_avg}</span>
								<span class="text-gray-500 dark:text-gray-400"
									>({template.rating_count || 0} {$i18n.t('ratings')})</span
								>
							</div>
						{/if}

						<!-- 사용 횟수 -->
						<div class="text-gray-600 dark:text-gray-400">
							{(template.usage_count || 0).toLocaleString()} {$i18n.t('uses')}
						</div>
					</div>

					<!-- 태그 -->
					{#if template.tags && template.tags.length > 0}
						<div class="flex flex-wrap gap-2 mb-6">
							{#each template.tags as tag}
								<span
									class="px-3 py-1 text-sm rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
								>
									{tag}
								</span>
							{/each}
						</div>
					{/if}

					<!-- 등록/삭제 버튼 -->
					<button
						on:click={handleToggle}
						class="px-6 py-3 font-semibold rounded-lg transition-colors duration-200 {isAdded
							? 'bg-red-600 hover:bg-red-700 text-white'
							: 'bg-blue-600 hover:bg-blue-700 text-white'}"
					>
						{isAdded ? $i18n.t('Remove from My GPTs') : $i18n.t('Add to My GPTs')}
					</button>
				</div>
			</div>
		</div>

		<!-- 상세 설명 -->
		{#if template.system_prompt}
			<div class="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-8 mb-6">
				<h2 class="text-xl font-semibold mb-4">{$i18n.t('About this GPT')}</h2>
				<div class="prose dark:prose-invert max-w-none">
					<pre class="whitespace-pre-wrap font-sans text-gray-700 dark:text-gray-300">{template.system_prompt}</pre>
				</div>
			</div>
		{/if}

		<!-- 대화 시작 예시 -->
		{#if template.conversation_starters && template.conversation_starters.length > 0}
			<div class="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-8">
				<h2 class="text-xl font-semibold mb-4">{$i18n.t('Conversation Starters')}</h2>
				<div class="space-y-2">
					{#each template.conversation_starters as starter}
						<div
							class="p-4 rounded-lg bg-gray-50 dark:bg-gray-700/50 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600"
						>
							"{starter}"
						</div>
					{/each}
				</div>
			</div>
		{/if}
	</div>
{:else}
	<!-- 로딩 상태 -->
	<div class="flex items-center justify-center h-full">
		<div class="text-gray-500 dark:text-gray-400">{$i18n.t('Loading...')}</div>
	</div>
{/if}
