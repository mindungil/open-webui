<script lang="ts">
    import { onMount } from 'svelte';
    import { getUsersWithUsage, getUserLimits, updateUserLimits, resetUserUsage } from '$lib/apis/usage';
    import { toast } from 'svelte-sonner';
    import { getContext } from 'svelte';
    const i18n = getContext('i18n');

    let users = [];
    let selectedUser = null;
    let userLimits = null;
    let loading = false;
    let editing = false;
    let showModal = false;
    let modalPosition = { x: 0, y: 0 };
    let editButtonRef: HTMLElement;

    async function loadUsers() {
        loading = true;
        try {
            const res = await getUsersWithUsage(localStorage.token);
            users = res.data.users;
        } catch (e) {
            toast.error($i18n.t('Failed to load user list'));
        }
        loading = false;
    }

    async function selectUser(user, event: MouseEvent) {
        selectedUser = user;
        editing = false;
        try {
            const res = await getUserLimits(localStorage.token, user.user_id);
            if (!res.success) {
                throw new Error(res.detail || 'Failed to load user limits');
            }
            userLimits = res.data;
            
            // Position modal to the left of the button
            const button = event.target as HTMLElement;
            const rect = button.getBoundingClientRect();
            modalPosition = {
                x: rect.left - 330, // Modal width (320px) + 10px gap
                y: rect.top
            };
            showModal = true;
        } catch (e) {
            console.error('Error loading user limits:', e);
            toast.error(e instanceof Error ? e.message : $i18n.t('Failed to load user limits'));
            showModal = false;
            selectedUser = null;
            userLimits = null;
        }
    }

    function closeModal() {
        showModal = false;
        selectedUser = null;
        userLimits = null;
    }

    async function saveLimits() {
        try {
            await updateUserLimits(localStorage.token, selectedUser.user_id, userLimits);
            toast.success($i18n.t('Limits saved successfully'));
            editing = false;
        } catch (e) {
            toast.error($i18n.t('Failed to save limits'));
        }
    }

    async function handleResetUsage() {
        if (!selectedUser) return;
        try {
            const res = await resetUserUsage(selectedUser.user_id, 'all');
            if (res.success) {
                alert('사용량이 성공적으로 리셋되었습니다.');
            } else {
                alert('사용량 리셋에 실패했습니다: ' + (res.message || res.detail));
            }
        } catch (e) {
            alert('에러 발생: ' + e);
        }
    }

    onMount(loadUsers);
</script>

<div class="container mx-auto p-4">
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 class="text-2xl font-bold mb-6 dark:text-white">{$i18n.t('')}</h2>
        
        {#if loading}
            <div class="flex justify-center items-center h-32">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 dark:border-white"></div>
            </div>
        {:else}
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead class="bg-gray-50 dark:bg-gray-700">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-large text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                {$i18n.t('사용자명')}
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-large text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                {$i18n.t('토큰 사용량(월)')}
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-large text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                {$i18n.t('요청 횟수(월)')}
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-large text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                {$i18n.t('마지막 활동')}
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-large text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                {$i18n.t('사용량 수정')}
                            </th>
                        </tr>
                    </thead>
                    <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                        {#each users as user}
                            <tr class="hover:bg-gray-50 dark:hover:bg-gray-700">
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-300">
                                    {user.user_name}
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-300">
                                    {user.monthly_tokens}
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-300">
                                    {user.monthly_requests}
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-300">
                                    {user.last_activity?.slice(0, 19).replace('T', ' ')}
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm">
                                    <button 
                                        class="inline-flex items-center px-3 py-1.5 text-sm font-medium text-indigo-600 bg-indigo-50 rounded-md hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200 dark:bg-indigo-900/30 dark:text-indigo-400 dark:hover:bg-indigo-900/50"
                                        on:click={(e) => selectUser(user, e)}
                                    >
                                        {$i18n.t('수정')}
                                    </button>
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {/if}
    </div>
</div>

{#if showModal && selectedUser && userLimits}
    <div 
        class="fixed z-50"
        style="left: {modalPosition.x}px; top: {modalPosition.y}px;"
    >
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-4 w-80 transform transition-all border border-gray-200 dark:border-gray-700">
            <div class="flex justify-between items-center mb-3">
                <h3 class="text-sm font-semibold dark:text-white">
                    {selectedUser.user_name} {$i18n.t('님 토큰 수정')}
                </h3>
                <button 
                    class="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-colors duration-200 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-600 whitespace-nowrap"
                    on:click={closeModal}
                >
                    {$i18n.t('닫기')}
                </button>
            </div>
            <div class="space-y-3">
                <div>
                    <label class="block text-xs font-medium text-gray-700 dark:text-gray-300">
                        {$i18n.t('일별 토큰 제한')}
                    </label>
                    <input 
                        type="number" 
                        bind:value={userLimits.daily_token_limit}
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white text-sm"
                    />
                </div>
                <div>
                    <label class="block text-xs font-medium text-gray-700 dark:text-gray-300">
                        {$i18n.t('월별 토큰 제한')}
                    </label>
                    <input 
                        type="number" 
                        bind:value={userLimits.monthly_token_limit}
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white text-sm"
                    />
                </div>
                <div>
                    <label class="block text-xs font-medium text-gray-700 dark:text-gray-300">
                        {$i18n.t('일별 요청 횟수 제한')}
                    </label>
                    <input 
                        type="number" 
                        bind:value={userLimits.daily_request_limit}
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white text-sm"
                    />
                </div>
                <div>
                    <label class="block text-xs font-medium text-gray-700 dark:text-gray-300">
                        {$i18n.t('월별 요청 횟수 제한')}
                    </label>
                    <input 
                        type="number" 
                        bind:value={userLimits.monthly_request_limit}
                        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white text-sm"
                    />
                </div>
                <div>
                    <label class="inline-flex items-center">
                        <input 
                            type="checkbox" 
                            bind:checked={userLimits.enabled}
                            class="rounded border-gray-300 text-indigo-600 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:border-gray-500"
                        />
                        <span class="ml-2 text-xs text-gray-700 dark:text-gray-300">
                            {$i18n.t('제한 활성화')}
                        </span>
                    </label>
                </div>
            </div>
            <div class="mt-4 flex justify-end space-x-2">
                <button 
                    class="inline-flex items-center px-3 py-1.5 text-sm font-medium text-rose-600 bg-rose-50 rounded-lg hover:bg-rose-100 focus:outline-none focus:ring-2 focus:ring-rose-500/20 focus:ring-offset-1 transition-all duration-200 dark:bg-rose-900/20 dark:text-rose-400 dark:hover:bg-rose-900/30 shadow-sm"
                    on:click={handleResetUsage}
                >
                    {$i18n.t('사용량 리셋')}
                </button>
                <button 
                    class="inline-flex items-center px-3 py-1.5 text-sm font-medium text-indigo-600 bg-indigo-50 rounded-md hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200 dark:bg-indigo-900/30 dark:text-indigo-400 dark:hover:bg-indigo-900/50"
                    on:click={saveLimits}
                >
                    {$i18n.t('저장')}
                </button>
            </div>
        </div>
    </div>
{/if}