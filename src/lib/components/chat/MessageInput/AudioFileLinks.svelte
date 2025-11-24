<script lang="ts">
	import { getContext } from 'svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let file: any = null;

	// Check if the file is an audio file (STT result)
	function isAudioFile(file: any): boolean {
		if (!file) return false;

		const fileName = file?.name ?? '';
		const fileType = file?.type ?? '';

		// Check by file extension
		const isAudioExtension = /\.(mp3|m4a|wav|flac|webm|aac)$/i.test(fileName);

		// Check by MIME type
		const isAudioType = fileType.startsWith('audio/') || fileType.startsWith('video/');

		return isAudioExtension || isAudioType;
	}

	// Decode URI string safely
	const decodeString = (str: string) => {
		try {
			return decodeURIComponent(str);
		} catch (e) {
			return str;
		}
	};

	// Replace audio file extension with new extension
	function replaceExtension(fileName: string, newExt: string): string {
		return fileName.replace(/\.(mp3|m4a|wav|flac|webm|aac|hwpx)$/i, newExt);
	}

	$: isAudio = isAudioFile(file);
	$: fileId = file?.file?.id ?? file?.id;
	$: fileName = file?.name ?? '';
</script>

{#if isAudio && fileId && file?.status === 'uploaded'}
	<div class="flex flex-col gap-1 pl-2 py-1">
		<Tooltip
			className="w-fit"
			content={$i18n.t('Download transcript (txt)')}
			placement="top-start"
			tippyOptions={{ duration: [500, 0] }}
		>
			<a
				class="text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 underline flex items-center gap-1"
				href={`${WEBUI_API_BASE_URL}/files/${fileId}txt/content?attachment=true`}
				target="_blank"
				on:click|stopPropagation
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
					class="size-3"
				>
					<path
						d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z"
					/>
					<path
						d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z"
					/>
				</svg>
				{decodeString(replaceExtension(fileName, '.txt'))}
			</a>
		</Tooltip>

		<Tooltip
			className="w-fit"
			content={$i18n.t('Download detailed transcript (docx)')}
			placement="top-start"
			tippyOptions={{ duration: [500, 0] }}
		>
			<a
				class="text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 underline flex items-center gap-1"
				href={`${WEBUI_API_BASE_URL}/files/${fileId}docx/content?attachment=true`}
				target="_blank"
				on:click|stopPropagation
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
					class="size-3"
				>
					<path
						d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z"
					/>
					<path
						d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z"
					/>
				</svg>
				{decodeString(replaceExtension(fileName, '.docx'))}
			</a>
		</Tooltip>

		<Tooltip
			className="w-fit"
			content={$i18n.t('Download detailed transcript (hwpx)')}
			placement="top-start"
			tippyOptions={{ duration: [500, 0] }}
		>
			<a
				class="text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 underline flex items-center gap-1"
				href={`${WEBUI_API_BASE_URL}/files/${fileId}hwpx/content?attachment=true`}
				target="_blank"
				on:click|stopPropagation
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
					class="size-3"
				>
					<path
						d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z"
					/>
					<path
						d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z"
					/>
				</svg>
				{decodeString(replaceExtension(fileName, '.hwpx'))}
			</a>
		</Tooltip>
	</div>
{/if}
