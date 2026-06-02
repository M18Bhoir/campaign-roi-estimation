/**
 * API client for Campaign ROI Estimation backend.
 * All requests go through this module — easy to swap base URL for prod/dev.
 */

const API_BASE = "http://localhost:8000/api/v1";

async function handleResponse(res) {
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "API request failed");
    }
    return res.json();
}

export const api = {
    async predictROI(payload) {
        const res = await fetch(`${API_BASE}/campaigns/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        return handleResponse(res);
    },

    async predictBatch(campaigns) {
        const res = await fetch(`${API_BASE}/campaigns/predict/batch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ campaigns })
        });
        return handleResponse(res);
    },

    async health() {
        const res = await fetch(`${API_BASE}/health`);
        return handleResponse(res);
    },

    async modelHealth() {
        const res = await fetch(`${API_BASE}/health/model`);
        return handleResponse(res);
    }
};
