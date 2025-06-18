import { WEBUI_API_BASE_URL } from '$lib/constants';

// 사용량 요약 정보 가져오기
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

// 사용자 목록과 사용량 가져오기
export async function getUsersWithUsage(token: string) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/users`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch users');
    return response.json();
}

// 특정 사용자의 사용량 가져오기
export async function getUserUsage(token: string, userId: string) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/user/${userId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch user usage');
    return response.json();
}

// 특정 사용자의 한도 가져오기
export async function getUserLimits(token: string, userId: string) {
    const response = await fetch(`${WEBUI_API_BASE_URL}/usage/limits/${userId}`, {
        headers: { 
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        }
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to fetch user limits');
    }
    
    const data = await response.json();
    if (!data.success) {
        throw new Error(data.detail || 'Failed to fetch user limits');
    }
    
    return data;
}

// 특정 사용자의 한도 수정하기
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

// 사용자 사용량 초기화하기
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