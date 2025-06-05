import { WEBUI_API_BASE_URL } from '$lib/constants';

export async function getUsageSummary(token: string) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/summary`, {
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        }
    });

    if (!response.ok) {
        throw new Error('Failed to fetch usage summary');
    }

    return response.json();
} 

// 기존 getUsageSummary 함수 아래에 추가
export async function getUsersWithUsage(token: string) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/users`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch users');
    return response.json();
}

export async function getUserUsage(token: string, userId: string) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/user/${userId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch user usage');
    return response.json();
}

export async function getUserLimits(token: string, userId: string) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/limits/${userId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch user limits');
    return response.json();
}

export async function updateUserLimits(token: string, userId: string, limits: any) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/limits/${userId}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(limits)
    });
    if (!response.ok) throw new Error('Failed to update user limits');
    return response.json();
}

export async function resetUserUsage(userId: string, resetType: string = 'all') {
    const token = localStorage.getItem('token');
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/reset/${userId}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ reset_type: resetType })
    });
    return response.json();
}