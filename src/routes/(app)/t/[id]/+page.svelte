<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';
	import { toast } from 'svelte-sonner';
	import { getContext, onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { get, type Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	import {
		user,
		config,
		settings,
		models,
		chatId,
		chats,
		currentChatPage,
		showSidebar,
		mobile
	} from '$lib/stores';

	import { createMessagesList, convertMessagesToHistory } from '$lib/utils';
	import { getGPTTemplateById } from '$lib/apis/gpt-templates';
	import { createNewChat, updateChatById, getChatList, getChatById } from '$lib/apis/chats';
	import { WEBUI_BASE_URL } from '$lib/constants';

	// 기존 컴포넌트 import
	import Messages from '$lib/components/chat/Messages.svelte';
	import MessageInput from '$lib/components/chat/MessageInput.svelte';
	import Sidebar from '$lib/components/icons/Sidebar.svelte';

	const i18n: Writable<i18nType> = getContext('i18n');

	let templateId = '';
	let template: any = null;
	let loading = true;

	let messageInput: any;
	let messagesContainerElement: HTMLDivElement;
	let autoScroll = true;

	let generating = false;
	let internalChatId = '';

	// Chat state
	let history = {
		messages: {},
		currentId: null
	};

	let prompt = '';
	let files: any[] = [];
	let chatFiles: any[] = [];

	// Dummy values for Messages/MessageInput compatibility
	let selectedModels = ['template'];
	let selectedToolIds: string[] = [];
	let selectedFilterIds: string[] = [];
	let imageGenerationEnabled = false;
	let webSearchEnabled = false;
	let codeInterpreterEnabled = false;
	let atSelectedModel: any = undefined;
	let showCommands = false;

	$: templateId = $page.params.id;
	$: existingChatId = $page.url.searchParams.get('chat');

	$: if (templateId || existingChatId) {
		loadTemplate();
	}

	const loadTemplate = async () => {
		loading = true;
		try {
			template = await getGPTTemplateById(localStorage.token, templateId);
			if (!template) {
				toast.error('템플릿을 찾을 수 없습니다');
				goto('/');
				return;
			}

			// 기존 채팅이 있으면 로드
			if (existingChatId) {
				const existingChat = await getChatById(localStorage.token, existingChatId);
				if (existingChat) {
					history = existingChat.chat.history || { messages: {}, currentId: null };
					internalChatId = existingChatId;
					chatId.set(existingChatId);
				} else {
					// 채팅을 찾을 수 없으면 새로 시작
					history = { messages: {}, currentId: null };
					internalChatId = '';
				}
			} else {
				// 새 채팅 시작
				history = { messages: {}, currentId: null };
				internalChatId = '';
			}

			prompt = '';
			files = [];
			chatFiles = [];

		} catch (error) {
			console.error('Failed to load template:', error);
			toast.error('템플릿 로딩 실패');
			goto('/');
		} finally {
			loading = false;
		}
	};

	const scrollToBottom = () => {
		if (messagesContainerElement) {
			messagesContainerElement.scrollTop = messagesContainerElement.scrollHeight;
		}
	};

	// 백엔드를 통해 템플릿 API 호출 (CORS 우회)
	const callTemplateAPI = async (messages: any[], chatId: string) => {
		if (!template?.api_url) {
			throw new Error('템플릿에 API URL이 설정되지 않았습니다');
		}

		// 시스템 프롬프트 추가
		const apiMessages = [];
		if (template.system_prompt) {
			apiMessages.push({
				role: 'system',
				content: template.system_prompt
			});
		}

		// 메시지 변환
		for (const msg of messages) {
			apiMessages.push({
				role: msg.role,
				content: msg.content
			});
		}

		// 백엔드의 chat completions 엔드포인트 호출
		const response = await fetch(`${WEBUI_BASE_URL}/api/chat/completions`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${localStorage.token}`
			},
			body: JSON.stringify({
				model: $models[0]?.id || 'gpt-4',
				messages: apiMessages,
				stream: true,
				chat_id: chatId
			})
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`API 오류: ${response.status} - ${errorText}`);
		}

		return response;
	};

	const submitPrompt = async (userPrompt: string) => {
		if (!userPrompt.trim() && files.length === 0) {
			toast.error('메시지를 입력해주세요');
			return;
		}

		if (!template?.api_url) {
			toast.error('템플릿에 API URL이 설정되지 않았습니다');
			return;
		}

		messageInput?.setText('');
		prompt = '';

		// Create user message
		const userMessageId = uuidv4();
		const userMessage = {
			id: userMessageId,
			parentId: history.currentId,
			childrenIds: [],
			role: 'user',
			content: userPrompt,
			files: files.length > 0 ? [...files] : undefined,
			timestamp: Math.floor(Date.now() / 1000)
		};

		// Add to history
		history.messages[userMessageId] = userMessage;
		if (history.currentId && history.messages[history.currentId]) {
			history.messages[history.currentId].childrenIds.push(userMessageId);
		}
		history.currentId = userMessageId;
		history = history;

		files = [];

		await sendMessage(userMessageId);
	};

	const sendMessage = async (userMessageId: string) => {
		generating = true;
		scrollToBottom();

		// Create assistant message placeholder
		const responseMessageId = uuidv4();
		const responseMessage = {
			id: responseMessageId,
			parentId: userMessageId,
			childrenIds: [],
			role: 'assistant',
			content: '',
			model: template?.name || 'Template',
			modelName: template?.name || 'Template',
			timestamp: Math.floor(Date.now() / 1000),
			done: false
		};

		history.messages[responseMessageId] = responseMessage;
		history.messages[userMessageId].childrenIds.push(responseMessageId);
		history.currentId = responseMessageId;
		history = history;

		await tick();
		scrollToBottom();

		try {
			// Create or update chat
			if (!internalChatId) {
				const chatData = {
					id: uuidv4(),
					title: template?.name || 'Template Chat',
					models: ['template'],
					messages: createMessagesList(history, responseMessageId),
					history: history,
					tags: [],
					timestamp: Date.now()
				};
				// createNewChat(token, chat, folderId, templateId)
				const chat = await createNewChat(localStorage.token, chatData, null, templateId);
				internalChatId = chat.id;
				chatId.set(chat.id);
			}

			// Get messages for API
			const messages = createMessagesList(history, userMessageId);

			// Call template API through backend
			const response = await callTemplateAPI(messages, internalChatId);

			// Handle streaming response
			const reader = response.body?.getReader();
			const decoder = new TextDecoder();

			if (reader) {
				let buffer = '';

				while (true) {
					const { done, value } = await reader.read();
					if (done) break;

					buffer += decoder.decode(value, { stream: true });
					const lines = buffer.split('\n');
					buffer = lines.pop() || '';

					for (const line of lines) {
						if (line.startsWith('data: ')) {
							const data = line.slice(6);
							if (data === '[DONE]') continue;

							try {
								const json = JSON.parse(data);
								const delta = json.choices?.[0]?.delta;

								// Handle text content
								const content = delta?.content || '';
								if (content) {
									history.messages[responseMessageId].content += content;
									history = history;
									scrollToBottom();
								}

								// Handle files in response
								const files = delta?.files || json.choices?.[0]?.message?.files;
								if (files && files.length > 0) {
									if (!history.messages[responseMessageId].files) {
										history.messages[responseMessageId].files = [];
									}
									history.messages[responseMessageId].files.push(...files);
									history = history;
									scrollToBottom();
								}
							} catch (e) {
								// Skip invalid JSON
							}
						}
					}
				}
			}

			// Mark as done
			history.messages[responseMessageId].done = true;
			history = history;

			// Save chat
			await updateChatById(localStorage.token, internalChatId, {
				messages: createMessagesList(history, history.currentId),
				history: history
			});

			// Refresh chat list
			currentChatPage.set(1);
			chats.set(await getChatList(localStorage.token, $currentChatPage));

		} catch (error: any) {
			console.error('API call failed:', error);
			history.messages[responseMessageId].content = `오류: ${error.message}`;
			history.messages[responseMessageId].done = true;
			history.messages[responseMessageId].error = true;
			history = history;
			toast.error(error.message);
		} finally {
			generating = false;
		}
	};

	const stopResponse = () => {
		generating = false;
	};

	// Handlers for Messages component (simplified versions)
	const showMessage = async (message: any) => {};

	const submitMessage = async (messageId: string, content: string) => {
		if (history.messages[messageId]) {
			history.messages[messageId].content = content;
			history = history;
		}
	};

	const continueResponse = async () => {
		if (history.currentId) {
			await sendMessage(history.currentId);
		}
	};

	const regenerateResponse = async (messageId?: string) => {
		const targetId = messageId || history.currentId;
		if (targetId && history.messages[targetId]) {
			const parentId = history.messages[targetId].parentId;
			if (parentId) {
				// Remove current response
				delete history.messages[targetId];
				history.currentId = parentId;
				history = history;
				await sendMessage(parentId);
			}
		}
	};

	const mergeResponses = async (messageId: string) => {};
	const chatActionHandler = async (action: string, messageId: string, event: any) => {};
	const addMessages = async (messages: any[]) => {};
	const createMessagePair = async (userMessage: any) => {};

	const onSelect = async (e: any) => {
		const { type, data } = e;
		if (type === 'prompt') {
			messageInput?.setText(data);
		}
	};
</script>

{#if !loading && template}
	<div class="h-screen max-h-[100dvh] w-full flex flex-col {$showSidebar ? 'md:max-w-[calc(100%-260px)]' : ''} md:ml-auto">
		<!-- Navbar -->
		<nav class="flex items-center justify-between px-4 py-2 border-b border-gray-100 dark:border-gray-850">
			<div class="flex items-center gap-2">
				<button
					class="p-1.5 hover:bg-gray-50 dark:hover:bg-gray-850 rounded-lg transition md:hidden"
					on:click={() => showSidebar.set(!$showSidebar)}
				>
					<Sidebar />
				</button>
				<a href="/" class="p-1.5 hover:bg-gray-50 dark:hover:bg-gray-850 rounded-lg transition">
					<svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
					</svg>
				</a>
			</div>

			<div class="flex items-center gap-2">
				{#if template.icon}
					<span class="text-xl">{template.icon}</span>
				{/if}
				<span class="font-medium truncate max-w-48">{template.name}</span>
			</div>

			<button
				class="p-2 hover:bg-gray-50 dark:hover:bg-gray-850 rounded-lg transition"
				on:click={loadTemplate}
				title="새 대화"
			>
				<svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
				</svg>
			</button>
		</nav>

		<!-- Messages area -->
		<div
			class="flex-1 overflow-auto pb-2.5"
			id="messages-container"
			bind:this={messagesContainerElement}
			on:scroll={() => {
				autoScroll = messagesContainerElement.scrollHeight - messagesContainerElement.scrollTop <= messagesContainerElement.clientHeight + 5;
			}}
		>
			{#if createMessagesList(history, history.currentId).length > 0}
				<div class="h-full w-full flex flex-col">
					<Messages
						chatId={internalChatId}
						bind:history
						bind:autoScroll
						bind:prompt
						setInputText={(text) => messageInput?.setText(text)}
						{selectedModels}
						{atSelectedModel}
						{sendMessage}
						{showMessage}
						{submitMessage}
						{continueResponse}
						{regenerateResponse}
						{mergeResponses}
						{chatActionHandler}
						{addMessages}
						topPadding={true}
						bottomPadding={files.length > 0}
						{onSelect}
					/>
				</div>
			{:else}
				<!-- Placeholder when no messages -->
				<div class="h-full flex flex-col items-center justify-center p-8">
					{#if template.icon}
						<div class="text-6xl mb-4">{template.icon}</div>
					{/if}
					<h2 class="text-xl font-semibold mb-2">{template.name}</h2>
					{#if template.description}
						<p class="text-gray-500 dark:text-gray-400 text-center max-w-md mb-6">
							{template.description}
						</p>
					{/if}

					<!-- Conversation starters -->
					{#if template.conversation_starters?.length > 0}
						<div class="flex flex-wrap gap-2 justify-center max-w-lg">
							{#each template.conversation_starters as starter}
								<button
									class="px-4 py-2 text-sm bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition"
									on:click={() => submitPrompt(starter)}
								>
									{starter}
								</button>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Input area -->
		<div class="pb-2 z-10">
			<MessageInput
				bind:this={messageInput}
				{history}
				taskIds={null}
				{selectedModels}
				bind:files
				bind:prompt
				bind:autoScroll
				bind:selectedToolIds
				bind:selectedFilterIds
				bind:imageGenerationEnabled
				bind:codeInterpreterEnabled
				bind:webSearchEnabled
				bind:atSelectedModel
				bind:showCommands
				toolServers={[]}
				{generating}
				{stopResponse}
				{createMessagePair}
				onChange={() => {}}
				on:upload={async (e) => {}}
				on:submit={async (e) => {
					if (e.detail) {
						await submitPrompt(e.detail.replaceAll('\n\n', '\n'));
					}
				}}
			/>
		</div>
	</div>
{:else}
	<div class="h-screen flex items-center justify-center">
		<div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
			<svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
			</svg>
			<span>로딩 중...</span>
		</div>
	</div>
{/if}
