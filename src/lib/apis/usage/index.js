import { WEBUI_API_BASE_URL } from '$lib/constants';

const getCurrentToken = () => {
	return localStorage.getItem('token') || '';
};

export const getUsageSummary = async (token = '') => {
	const authToken = token || getCurrentToken();
	
	if (!authToken) {
		throw new Error('No token available');
	}

	const res = await fetch(`${WEBUI_API_BASE_URL}/usage/summary`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `Bearer ${authToken}`
		}
	});

	if (!res.ok) {
		throw new Error(`HTTP ${res.status}`);
	}

	return res.json();
};
