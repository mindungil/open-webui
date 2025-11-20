<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { user } from '$lib/stores';
	import {
		getGPTTemplates,
		createGPTTemplate,
		updateGPTTemplate,
		deleteGPTTemplate
	} from '$lib/apis/gpt-templates';

	const i18n = getContext('i18n');

	let templates = [];
	let loaded = false;
	let showModal = false;
	let editingTemplate = null;
	let formData = {
		name: '',
		description: '',
		category: '',
		icon: '💬',
		tags: [] as string[],
		system_prompt: '',
		api_url: '',
		is_public: true,
		is_featured: false
	};
	let tagInput = '';

	const loadTemplates = async () => {
		try {
			templates = await getGPTTemplates(localStorage.token);
			loaded = true;
		} catch (error) {
			console.error('Failed to load templates:', error);
			toast.error('템플릿 목록을 불러오는데 실패했습니다.');
		}
	};

	const openCreateModal = () => {
		editingTemplate = null;
		formData = {
			name: '',
			description: '',
			category: '',
			icon: '💬',
			tags: [],
			system_prompt: '',
			api_url: '',
			is_public: true,
			is_featured: false
		};
		tagInput = '';
		showModal = true;
	};

	const openEditModal = (template: any) => {
		editingTemplate = template;
		formData = {
			name: template.name || '',
			description: template.description || '',
			category: template.category || '',
			icon: template.icon || '💬',
			tags: template.tags || [],
			system_prompt: template.system_prompt || '',
			api_url: template.api_url || '',
			is_public: template.is_public ?? true,
			is_featured: template.is_featured ?? false
		};
		tagInput = '';
		showModal = true;
	};

	const addTag = () => {
		if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
			formData.tags = [...formData.tags, tagInput.trim()];
			tagInput = '';
		}
	};

	const removeTag = (tag: string) => {
		formData.tags = formData.tags.filter((t) => t !== tag);
	};

	const saveTemplate = async () => {
		try {
			if (!formData.name.trim()) {
				toast.error('템플릿 이름을 입력해주세요.');
				return;
			}

			const templateData = {
				...formData
			};

			if (editingTemplate) {
				await updateGPTTemplate(localStorage.token, editingTemplate.id, templateData);
				toast.success('템플릿이 수정되었습니다.');
			} else {
				await createGPTTemplate(localStorage.token, templateData);
				toast.success('템플릿이 생성되었습니다.');
			}

			showModal = false;
			await loadTemplates();
		} catch (error: any) {
			console.error('Failed to save template:', error);
			toast.error(error?.detail || '템플릿 저장에 실패했습니다.');
		}
	};

	const handleDelete = async (id: string, name: string) => {
		if (!confirm(`"${name}" 템플릿을 삭제하시겠습니까?`)) {
			return;
		}

		try {
			await deleteGPTTemplate(localStorage.token, id);
			toast.success('템플릿이 삭제되었습니다.');
			await loadTemplates();
		} catch (error: any) {
			console.error('Failed to delete template:', error);
			toast.error(error?.detail || '템플릿 삭제에 실패했습니다.');
		}
	};

	onMount(() => {
		if ($user?.role === 'admin') {
			loadTemplates();
		}
	});
</script>

<div class="flex flex-col h-full p-6">
	<div class="flex items-center justify-between mb-6">
		<div>
			<h1 class="text-2xl font-semibold">템플릿 관리</h1>
			<p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
				GPT 템플릿을 생성, 수정, 삭제할 수 있습니다.
			</p>
		</div>
		<button
			on:click={openCreateModal}
			class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
		>
			+ 새 템플릿
		</button>
	</div>

	{#if loaded}
		<div class="flex-1 overflow-y-auto">
			{#if templates.length > 0}
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
					{#each templates as template}
						<div
							class="p-5 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:shadow-lg transition"
						>
							<div class="flex items-start gap-3 mb-3">
								<div class="text-4xl">{template.icon || '💬'}</div>
								<div class="flex-1 min-w-0">
									<div class="font-semibold text-lg truncate">{template.name}</div>
									<div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
										{template.usage_count || 0}회 사용
									</div>
								</div>
							</div>

							<div class="text-sm text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
								{template.description || '설명 없음'}
							</div>

							{#if template.api_url}
								<div class="text-xs text-gray-500 dark:text-gray-400 mb-3 truncate">
									🔗 {template.api_url}
								</div>
							{/if}

							<div class="flex gap-2 mt-4">
								<button
									on:click={() => openEditModal(template)}
									class="flex-1 px-3 py-2 text-sm bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition"
								>
									수정
								</button>
								<button
									on:click={() => handleDelete(template.id, template.name)}
									class="flex-1 px-3 py-2 text-sm bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-200 dark:hover:bg-red-800 transition"
								>
									삭제
								</button>
							</div>
						</div>
					{/each}
				</div>
			{:else}
				<div class="text-center py-12">
					<div class="text-gray-500 dark:text-gray-400">템플릿이 없습니다.</div>
				</div>
			{/if}
		</div>
	{:else}
		<div class="flex items-center justify-center h-full">
			<div class="text-gray-500 dark:text-gray-400">로딩 중...</div>
		</div>
	{/if}
</div>

<!-- 템플릿 생성/수정 모달 -->
{#if showModal}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
		on:click={() => (showModal = false)}
		role="button"
		tabindex="0"
	>
		<div
			class="bg-white dark:bg-gray-800 rounded-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto"
			on:click|stopPropagation
		>
			<h2 class="text-xl font-semibold mb-4">
				{editingTemplate ? '템플릿 수정' : '새 템플릿 생성'}
			</h2>

			<div class="space-y-4">
				<div>
					<label class="block text-sm font-medium mb-1">템플릿 이름 *</label>
					<input
						type="text"
						bind:value={formData.name}
						class="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800"
						placeholder="템플릿 이름"
					/>
				</div>

				<div>
					<label class="block text-sm font-medium mb-1">설명</label>
					<textarea
						bind:value={formData.description}
						class="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800"
						rows="3"
						placeholder="템플릿 설명"
					></textarea>
				</div>

				<div class="grid grid-cols-2 gap-4">
					<div>
						<label class="block text-sm font-medium mb-1">카테고리</label>
						<input
							type="text"
							bind:value={formData.category}
							class="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800"
							placeholder="카테고리"
						/>
					</div>

					<div>
						<label class="block text-sm font-medium mb-1">아이콘</label>
						<input
							type="text"
							bind:value={formData.icon}
							class="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800"
							placeholder="💬"
						/>
					</div>
				</div>

				<div>
					<label class="block text-sm font-medium mb-1">태그</label>
					<div class="flex gap-2 mb-2">
						<input
							type="text"
							bind:value={tagInput}
							on:keydown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
							class="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800"
							placeholder="태그 입력 후 Enter"
						/>
						<button
							on:click={addTag}
							class="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
						>
							추가
						</button>
					</div>
					<div class="flex flex-wrap gap-2">
						{#each formData.tags as tag}
							<span
								class="px-2 py-1 text-xs rounded-full bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 flex items-center gap-1"
							>
								{tag}
								<button
									on:click={() => removeTag(tag)}
									class="hover:text-blue-900 dark:hover:text-blue-100"
								>
									×
								</button>
							</span>
						{/each}
					</div>
				</div>

				<div>
					<label class="block text-sm font-medium mb-1">시스템 프롬프트</label>
					<textarea
						bind:value={formData.system_prompt}
						class="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800"
						rows="4"
						placeholder="시스템 프롬프트"
					></textarea>
				</div>

				<div class="border-t border-gray-200 dark:border-gray-700 pt-4">
					<h3 class="text-lg font-medium mb-3">API 연동 설정</h3>
					<div>
						<label class="block text-sm font-medium mb-1">API URL (포트 포함)</label>
						<input
							type="text"
							bind:value={formData.api_url}
							class="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800"
							placeholder="http://example.com:8080/v1"
						/>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
							전체 API 엔드포인트를 입력하세요. /chat/completions는 자동으로 추가됩니다.
						</p>
					</div>
				</div>

				<div class="flex gap-4">
					<label class="flex items-center gap-2">
						<input type="checkbox" bind:checked={formData.is_public} />
						<span class="text-sm">공개</span>
					</label>
					<label class="flex items-center gap-2">
						<input type="checkbox" bind:checked={formData.is_featured} />
						<span class="text-sm">추천</span>
					</label>
				</div>
			</div>

			<div class="flex gap-2 mt-6">
				<button
					on:click={saveTemplate}
					class="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
				>
					{editingTemplate ? '수정' : '생성'}
				</button>
				<button
					on:click={() => (showModal = false)}
					class="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition"
				>
					취소
				</button>
			</div>
		</div>
	</div>
{/if}
