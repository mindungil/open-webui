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

    async function selectUser(user) {
        selectedUser = user;
        editing = false;
        try {
            const res = await getUserLimits(localStorage.token, user.user_id);
            userLimits = res.data;
        } catch (e) {
            toast.error($i18n.t('Failed to load user limits'));
        }
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
        <h2 class="text-2xl font-bold mb-6 dark:text-white">{$i18n.t('전체 사용자 토큰 관리')}</h2>
        
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
                                {$i18n.t('비고')}
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
                                        class="text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300"
                                        on:click={() => selectUser(user)}
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

        {#if selectedUser && userLimits}
            <div class="mt-8 p-6 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <h3 class="text-lg font-semibold mb-4 dark:text-white">
                    {selectedUser.user_name} {$i18n.t('님 토큰 수정')}
                </h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                {$i18n.t('일별 토큰 제한')}
                            </label>
                            <input 
                                type="number" 
                                bind:value={userLimits.daily_token_limit}
                                class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white"
                            />
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                {$i18n.t('월별 토큰 제한')}
                            </label>
                            <input 
                                type="number" 
                                bind:value={userLimits.monthly_token_limit}
                                class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white"
                            />
                        </div>
                    </div>
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                {$i18n.t('일별 요청 횟수 제한')}
                            </label>
                            <input 
                                type="number" 
                                bind:value={userLimits.daily_request_limit}
                                class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white"
                            />
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                {$i18n.t('월별 요청 횟수 제한')}
                            </label>
                            <input 
                                type="number" 
                                bind:value={userLimits.monthly_request_limit}
                                class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-600 dark:border-gray-500 dark:text-white"
                            />
                        </div>
                    </div>
                </div>
                <div class="mt-4">
                    <label class="inline-flex items-center">
                        <input 
                            type="checkbox" 
                            bind:checked={userLimits.enabled}
                            class="rounded border-gray-300 text-indigo-600 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:border-gray-500"
                        />
                        <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">
                            {$i18n.t('제한 활성화')}
                        </span>
                    </label>
                </div>
                <div class="mt-6 flex justify-end space-x-3">
                    <button 
                        class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 dark:bg-gray-600 dark:text-gray-300 dark:border-gray-500 dark:hover:bg-gray-500"
                        on:click={() => { selectedUser = null; userLimits = null; }}
                    >
                        {$i18n.t('취소')}
                    </button>
                    <button 
                        class="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                        on:click={handleResetUsage}
                    >
                        {$i18n.t('사용량 리셋')}
                    </button>
                    <button 
                        class="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        on:click={saveLimits}
                    >
                        {$i18n.t('저장')}
                    </button>
                </div>
            </div>
        {/if}
    </div>
</div>