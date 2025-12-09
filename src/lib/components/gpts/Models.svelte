<script lang="ts">
	import { marked } from 'marked';

	import { toast } from 'svelte-sonner';
	import Sortable from 'sortablejs';

	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { onMount, getContext, tick } from 'svelte';
	import { goto } from '$app/navigation';
	const i18n = getContext('i18n');

	import { WEBUI_NAME, config, mobile, models as _models, settings, user } from '$lib/stores';
	import { WEBUI_BASE_URL } from '$lib/constants';
	import {
		createNewModel,
		deleteModelById,
		getModelItems as getWorkspaceModels,
		toggleModelById,
		updateModelById
	} from '$lib/apis/models';

	import { getModels } from '$lib/apis';
	import { getGroups } from '$lib/apis/groups';

	import { capitalizeFirstLetter, copyToClipboard } from '$lib/utils';

	import EllipsisHorizontal from '../icons/EllipsisHorizontal.svelte';
	import ModelMenu from './Models/ModelMenu.svelte';
	import ModelDeleteConfirmDialog from '../common/ConfirmDialog.svelte';
	import Tooltip from '../common/Tooltip.svelte';
	import GarbageBin from '../icons/GarbageBin.svelte';
	import Search from '../icons/Search.svelte';
	import Plus from '../icons/Plus.svelte';
	import ChevronRight from '../icons/ChevronRight.svelte';
	import Switch from '../common/Switch.svelte';
	import Spinner from '../common/Spinner.svelte';
	import XMark from '../icons/XMark.svelte';
	import EyeSlash from '../icons/EyeSlash.svelte';
	import Eye from '../icons/Eye.svelte';
	import ViewSelector from './common/ViewSelector.svelte';
	import TagSelector from './common/TagSelector.svelte';

	let shiftKey = false;

	let importFiles;
	let modelsImportInputElement: HTMLInputElement;
	let tagsContainerElement: HTMLDivElement;

	let loaded = false;

	let models = [];
	let tags = [];

	let viewOption = '';
	let selectedTag = '';

	let filteredModels = [];
	let selectedModel = null;

	let showModelDeleteConfirm = false;

	let group_ids = [];
	let groups = [];

	$: if (models && query !== undefined && selectedTag !== undefined && viewOption !== undefined) {
		setFilteredModels();
	}

	// Helper function to get group names from group IDs
	const getGroupNames = (groupIds) => {
		if (!groupIds || groupIds.length === 0) return [];
		return groupIds
			.map((gid) => groups.find((g) => g.id === gid))
			.filter((g) => g)
			.map((g) => g.name);
	};

	const setFilteredModels = async () => {
		filteredModels = models.filter((m) => {
			if (query === '' && selectedTag === '' && viewOption === '') return true;
			const lowerQuery = query.toLowerCase();
			return (
				((m.name || '').toLowerCase().includes(lowerQuery) ||
					(m.user?.name || '').toLowerCase().includes(lowerQuery) || // Search by user name
					(m.user?.email || '').toLowerCase().includes(lowerQuery)) && // Search by user email
				(selectedTag === '' ||
					m?.meta?.tags?.some((tag) => tag.name.toLowerCase() === selectedTag.toLowerCase())) &&
				(viewOption === '' ||
					(viewOption === 'created' && m.user_id === $user?.id) ||
					(viewOption === 'shared' && m.user_id !== $user?.id))
			);
		});
	};

	let query = '';
	const deleteModelHandler = async (model) => {
		const res = await deleteModelById(localStorage.token, model.id).catch((e) => {
			toast.error(`${e}`);
			return null;
		});

		if (res) {
			toast.success($i18n.t(`Deleted {{name}}`, { name: model.id }));
		}

		await _models.set(
			await getModels(
				localStorage.token,
				$config?.features?.enable_direct_connections && ($settings?.directConnections ?? null)
			)
		);
		models = await getWorkspaceModels(localStorage.token, 'gpts');
	};

	const cloneModelHandler = async (model) => {
		sessionStorage.model = JSON.stringify({
			...model,
			id: `${model.id}-clone`,
			name: `${model.name} (Clone)`
		});
		goto('/gpts/models/create');
	};

	const shareModelHandler = async (model) => {
		toast.success($i18n.t('Redirecting you to Open WebUI Community'));

		const url = 'https://openwebui.com';

		const tab = await window.open(`${url}/models/create`, '_blank');

		const messageHandler = (event) => {
			if (event.origin !== url) return;
			if (event.data === 'loaded') {
				tab.postMessage(JSON.stringify(model), '*');
				window.removeEventListener('message', messageHandler);
			}
		};

		window.addEventListener('message', messageHandler, false);
	};

	const hideModelHandler = async (model) => {
		model.meta = {
			...model.meta,
			hidden: !(model?.meta?.hidden ?? false)
		};

		console.log(model);

		const res = await updateModelById(localStorage.token, model.id, model);

		if (res) {
			toast.success(
				$i18n.t(`Model {{name}} is now {{status}}`, {
					name: model.id,
					status: model.meta.hidden ? 'hidden' : 'visible'
				})
			);
		}

		await _models.set(
			await getModels(
				localStorage.token,
				$config?.features?.enable_direct_connections && ($settings?.directConnections ?? null)
			)
		);
		models = await getWorkspaceModels(localStorage.token, 'gpts');
	};

	const copyLinkHandler = async (model) => {
		const baseUrl = window.location.origin;
		const res = await copyToClipboard(`${baseUrl}/?model=${encodeURIComponent(model.id)}`);

		if (res) {
			toast.success($i18n.t('Copied link to clipboard'));
		} else {
			toast.error($i18n.t('Failed to copy link'));
		}
	};

	const downloadModels = async (models) => {
		let blob = new Blob([JSON.stringify(models)], {
			type: 'application/json'
		});
		saveAs(blob, `models-export-${Date.now()}.json`);
	};

	const exportModelHandler = async (model) => {
		let blob = new Blob([JSON.stringify([model])], {
			type: 'application/json'
		});
		saveAs(blob, `${model.id}-${Date.now()}.json`);
	};

	const setTags = () => {
		if (models) {
			tags = models
				.filter((model) => !(model?.meta?.hidden ?? false))
				.flatMap((model) => model?.meta?.tags ?? [])
				.map((tag) => tag.name);

			// Remove duplicates and sort
			tags = Array.from(new Set(tags)).sort((a, b) => a.localeCompare(b));
		}
	};

	onMount(async () => {
		viewOption = localStorage.workspaceViewOption ?? '';

		models = await getWorkspaceModels(localStorage.token, 'gpts');
		groups = await getGroups(localStorage.token);
		group_ids = groups.map((group) => group.id);

		setTags();
		loaded = true;

		const onKeyDown = (event) => {
			if (event.key === 'Shift') {
				shiftKey = true;
			}
		};

		const onKeyUp = (event) => {
			if (event.key === 'Shift') {
				shiftKey = false;
			}
		};

		const onBlur = () => {
			shiftKey = false;
		};

		window.addEventListener('keydown', onKeyDown);
		window.addEventListener('keyup', onKeyUp);
		window.addEventListener('blur-sm', onBlur);

		return () => {
			window.removeEventListener('keydown', onKeyDown);
			window.removeEventListener('keyup', onKeyUp);
			window.removeEventListener('blur-sm', onBlur);
		};
	});
</script>

<svelte:head>
	<title>
		{$i18n.t('Models')} • {$WEBUI_NAME}
	</title>
</svelte:head>

{#if loaded}
	<ModelDeleteConfirmDialog
		bind:show={showModelDeleteConfirm}
		on:confirm={() => {
			deleteModelHandler(selectedModel);
		}}
	/>

	<div class="flex flex-col gap-1 px-1 mt-1.5 mb-3">
		<input
			id="models-import-input"
			bind:this={modelsImportInputElement}
			bind:files={importFiles}
			type="file"
			accept=".json"
			hidden
			on:change={() => {
				console.log(importFiles);

				let reader = new FileReader();
				reader.onload = async (event) => {
					let savedModels = JSON.parse(event.target.result);
					console.log(savedModels);

					for (const model of savedModels) {
						if (model?.info ?? false) {
							if ($_models.find((m) => m.id === model.id)) {
								await updateModelById(localStorage.token, model.id, model.info).catch((error) => {
									return null;
								});
							} else {
								await createNewModel(localStorage.token, model.info).catch((error) => {
									return null;
								});
							}
						} else {
							if (model?.id && model?.name) {
								await createNewModel(localStorage.token, model).catch((error) => {
									return null;
								});
							}
						}
					}

					await _models.set(
						await getModels(
							localStorage.token,
							$config?.features?.enable_direct_connections && ($settings?.directConnections ?? null)
						)
					);
					models = await getWorkspaceModels(localStorage.token, 'gpts');
				};

				reader.readAsText(importFiles[0]);
			}}
		/>
		<div class="mb-6 pt-2">
			<div class="flex justify-between items-center mb-4">
				<div>
					<h1 class="text-3xl font-bold tracking-tight text-gray-900 dark:text-white mb-1">
						GPTs
					</h1>
					<p class="text-sm text-gray-500 dark:text-gray-400">
						사용자가 만든 GPT
					</p>
				</div>

			<div class="flex w-full justify-end gap-1.5">
				{#if $user?.role === 'admin'}
					<button
						class="flex text-xs items-center space-x-1 px-3 py-1.5 rounded-xl bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-gray-200 transition"
						on:click={() => {
							modelsImportInputElement.click();
						}}
					>
						<div class=" self-center font-medium line-clamp-1">
							{$i18n.t('Import')}
						</div>
					</button>

					{#if models.length}
						<button
							class="flex text-xs items-center space-x-1 px-3 py-1.5 rounded-xl bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-gray-200 transition"
							on:click={async () => {
								downloadModels(models);
							}}
						>
							<div class=" self-center font-medium line-clamp-1">
								{$i18n.t('Export')}
							</div>
						</button>
					{/if}
				{/if}
				<a
					class="px-4 py-2.5 rounded-xl bg-gray-900 hover:bg-gray-800 dark:bg-white dark:hover:bg-gray-100 text-white dark:text-gray-900 transition font-semibold text-sm flex items-center gap-2 shadow-sm"
					href="/gpts/models/create"
				>
					<Plus className="size-4" strokeWidth="2.5" />
					<div class="hidden md:block text-sm">새 GPT 만들기</div>
				</a>
			</div>
		</div>
	</div>

	<div class="space-y-4">
		<div class="px-4 flex flex-1 items-center w-full">
			<div class="flex flex-1 items-center bg-transparent rounded-xl px-4 py-3 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition">
				<div class="self-center mr-3">
					<Search className="size-4 text-gray-400" />
				</div>
				<input
					class="w-full text-sm outline-hidden bg-transparent placeholder-gray-400 dark:placeholder-gray-500"
					bind:value={query}
					placeholder="GPTs 검색..."
				/>

				{#if query}
					<div class="self-center pl-1.5">
						<button
							class="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition"
							on:click={() => {
								query = '';
							}}
						>
							<XMark className="size-4 text-gray-400" strokeWidth="2" />
						</button>
					</div>
				{/if}
			</div>
		</div>

		<div
			class="px-3 flex w-full bg-transparent overflow-x-auto scrollbar-none"
			on:wheel={(e) => {
				if (e.deltaY !== 0) {
					e.preventDefault();
					e.currentTarget.scrollLeft += e.deltaY;
				}
			}}
		>
			<div
				class="flex gap-0.5 w-fit text-center text-sm rounded-full bg-transparent px-0.5 whitespace-nowrap"
				bind:this={tagsContainerElement}
			>
				<ViewSelector
					bind:value={viewOption}
					onChange={async (value) => {
						localStorage.workspaceViewOption = value;

						await tick();
						setTags();
					}}
				/>

				{#if (tags ?? []).length > 0}
					<TagSelector
						bind:value={selectedTag}
						items={tags.map((tag) => {
							return { value: tag, label: tag };
						})}
					/>
				{/if}
			</div>
		</div>

		{#if (filteredModels ?? []).length !== 0}
			<div class=" px-4 my-3 gap-3 lg:gap-4 grid lg:grid-cols-2" id="model-list">
				{#each filteredModels as model (model.id)}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<!-- svelte-ignore a11y_click_events_have_key_events -->
					<div
						class="group flex cursor-pointer hover:bg-gray-50/50 dark:hover:bg-gray-800/50 transition rounded-2xl w-full p-5 border border-gray-200 dark:border-gray-700 bg-transparent"
						id="model-item-{model.id}"
						on:click={() => {
							if (
								$user?.role === 'admin' ||
								model.user_id === $user?.id ||
								model.access_control.write.group_ids.some((wg) => group_ids.includes(wg))
							) {
								goto(`/gpts/models/edit?id=${encodeURIComponent(model.id)}`);
							}
						}}
					>
						<div class="flex group/item gap-4 w-full">
							<div class="self-center">
								<div class="relative">
									<img
										src={model?.meta?.profile_image_url ?? `${WEBUI_BASE_URL}/static/favicon.png`}
										alt="modelfile profile"
										class="rounded-2xl size-16 object-cover {model.is_active ? '' : 'opacity-40 grayscale'}"
									/>
									{#if model.is_active}
										<div class="absolute -bottom-0.5 -right-0.5 size-3.5 bg-green-500 rounded-full border-2 border-white dark:border-gray-950"></div>
									{/if}
								</div>
							</div>

							<div class=" shrink-0 flex w-full min-w-0 flex-1 pr-1 self-center">
								<div class="flex h-full w-full flex-1 flex-col justify-start self-center group">
									<div class="flex-1 w-full">
										<div class="flex items-center justify-between w-full">
											<Tooltip content={model.name} className=" w-fit" placement="top-start">
												<a
													class="text-base font-semibold line-clamp-1 hover:underline capitalize text-gray-900 dark:text-white"
													href={`/?models=${encodeURIComponent(model.id)}`}
												>
													{model.name}
												</a>
											</Tooltip>

											<div class=" flex items-center gap-1">
												<div
													class="flex justify-end w-full {model.is_active ? '' : 'text-gray-500'}"
												>
													<div class="flex justify-between items-center w-full">
														<div class=""></div>
														<div class="flex flex-row gap-0.5 items-center">
															{#if shiftKey}
																<Tooltip
																	content={model?.meta?.hidden ? $i18n.t('Show') : $i18n.t('Hide')}
																>
																	<button
																		class="self-center w-fit text-sm p-1.5 dark:text-white hover:bg-black/5 dark:hover:bg-white/5 rounded-xl"
																		type="button"
																		on:click={(e) => {
																			e.stopPropagation();
																			hideModelHandler(model);
																		}}
																	>
																		{#if model?.meta?.hidden}
																			<EyeSlash />
																		{:else}
																			<Eye />
																		{/if}
																	</button>
																</Tooltip>

																<Tooltip content={$i18n.t('Delete')}>
																	<button
																		class="self-center w-fit text-sm p-1.5 dark:text-white hover:bg-black/5 dark:hover:bg-white/5 rounded-xl"
																		type="button"
																		on:click={(e) => {
																			e.stopPropagation();
																			deleteModelHandler(model);
																		}}
																	>
																		<GarbageBin />
																	</button>
																</Tooltip>
															{:else}
																<ModelMenu
																	user={$user}
																	{model}
																	editHandler={() => {
																		goto(
																			`/gpts/models/edit?id=${encodeURIComponent(model.id)}`
																		);
																	}}
																	shareHandler={() => {
																		shareModelHandler(model);
																	}}
																	cloneHandler={() => {
																		cloneModelHandler(model);
																	}}
																	exportHandler={() => {
																		exportModelHandler(model);
																	}}
																	hideHandler={() => {
																		hideModelHandler(model);
																	}}
																	copyLinkHandler={() => {
																		copyLinkHandler(model);
																	}}
																	deleteHandler={() => {
																		selectedModel = model;
																		showModelDeleteConfirm = true;
																	}}
																	onClose={() => {}}
																>
																	<div
																		class="self-center w-fit p-1 text-sm dark:text-white hover:bg-black/5 dark:hover:bg-white/5 rounded-xl"
																	>
																		<EllipsisHorizontal className="size-5" />
																	</div>
																</ModelMenu>
															{/if}
														</div>
													</div>
												</div>

												<div class="flex items-center gap-2">
													<button
														on:click={(e) => {
															e.stopPropagation();
														}}
													>
														<Tooltip
															content={model.is_active ? $i18n.t('Enabled') : $i18n.t('Disabled')}
														>
															<Switch
																bind:state={model.is_active}
																on:change={async () => {
																	toggleModelById(localStorage.token, model.id);
																	_models.set(
																		await getModels(
																			localStorage.token,
																			$config?.features?.enable_direct_connections &&
																				($settings?.directConnections ?? null)
																		)
																	);
																}}
															/>
														</Tooltip>
													</button>
												</div>
											</div>
										</div>

										<div class=" flex flex-col gap-1 pr-2 -mt-0.5">
											<div class="flex gap-1.5 items-center flex-wrap">
												<Tooltip
													content={model?.user?.email ?? $i18n.t('Deleted User')}
													className="flex shrink-0"
													placement="top-start"
												>
													<div class="shrink-0 text-gray-600 dark:text-gray-400 text-xs font-medium">
														{$i18n.t('By {{name}}', {
															name: capitalizeFirstLetter(
																model?.user?.name ?? model?.user?.email ?? $i18n.t('Deleted User')
															)
														})}
													</div>
												</Tooltip>

												{#if model?.access_control?.write?.group_ids && model.access_control.write.group_ids.length > 0}
													{@const groupNames = getGroupNames(model.access_control.write.group_ids)}
													{#if groupNames.length > 0}
														<div class="text-gray-400 dark:text-gray-600">·</div>
														<div class="flex gap-1 items-center flex-wrap">
															{#each groupNames as groupName, idx}
																<span
																	class="inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
																>
																	<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3 mr-1">
																		<path stroke-linecap="round" stroke-linejoin="round" d="M18 18.72a9.094 9.094 0 0 0 3.741-.479 3 3 0 0 0-4.682-2.72m.94 3.198.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0 1 12 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 0 1 6 18.719m12 0a5.971 5.971 0 0 0-.941-3.197m0 0A5.995 5.995 0 0 0 12 12.75a5.995 5.995 0 0 0-5.058 2.772m0 0a3 3 0 0 0-4.681 2.72 8.986 8.986 0 0 0 3.74.477m.94-3.197a5.971 5.971 0 0 0-.94 3.197M15 6.75a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm6 3a2.25 2.25 0 1 1-4.5 0 2.25 2.25 0 0 1 4.5 0Zm-13.5 0a2.25 2.25 0 1 1-4.5 0 2.25 2.25 0 0 1 4.5 0Z" />
																	</svg>
																	{groupName}
																</span>
															{/each}
														</div>
													{/if}
												{/if}
											</div>

											<Tooltip
												content={marked.parse(model?.meta?.description ?? model.id)}
												className=" w-fit text-left"
												placement="top-start"
											>
												<div class="flex gap-1 text-sm overflow-hidden text-gray-600 dark:text-gray-400 mt-1">
													<div class="line-clamp-2 leading-relaxed">
														{#if (model?.meta?.description ?? '').trim()}
															{model?.meta?.description}
														{:else}
															<span class="text-gray-400 dark:text-gray-500 italic">{model.id}</span>
														{/if}
													</div>
												</div>
											</Tooltip>
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				{/each}
			</div>
		{:else}
			<div class="w-full h-full flex flex-col justify-center items-center my-24 px-4">
				<div class="max-w-md text-center">
					<div class="mb-6">
						<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-16 mx-auto text-gray-300 dark:text-gray-700">
							<path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
						</svg>
					</div>
					<h3 class="text-lg font-semibold mb-2 text-gray-900 dark:text-white">GPT를 찾을 수 없습니다</h3>
					<p class="text-gray-500 dark:text-gray-400 text-sm mb-6">
						검색어나 필터를 조정하여 원하는 GPT를 찾아보세요
					</p>
					<a
						href="/gpts/models/create"
						class="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-900 hover:bg-gray-800 dark:bg-white dark:hover:bg-gray-100 text-white dark:text-gray-900 transition font-semibold text-sm shadow-sm"
					>
						<Plus className="size-4" strokeWidth="2.5" />
						새 GPT 만들기
					</a>
				</div>
			</div>
		{/if}
	</div>
</div>

{:else}
	<div class="w-full h-full flex justify-center items-center">
		<Spinner className="size-5" />
	</div>
{/if}
