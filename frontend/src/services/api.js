import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

/* ============================================================
   MODEL METADATA
   ============================================================ */

export const getModelMetadata = async () => {
  const response = await api.get("/api/model/metadata");

  return response.data;
};

/* ============================================================
   ROI PREDICTION
   ============================================================ */

export const predictROI = async (payload) => {
  const response = await api.post("/api/predict", payload);

  return response.data;
};

/* ============================================================
   PREDICTION HISTORY
   ============================================================ */

export const getPredictionHistory = async ({
  limit = 20,
  offset = 0,
  campaignType = null,
  marketingChannel = null,
  modelVersion = null,
} = {}) => {
  const params = {
    limit,
    offset,
  };

  if (campaignType) {
    params.campaign_type = campaignType;
  }

  if (marketingChannel) {
    params.marketing_channel = marketingChannel;
  }

  if (modelVersion) {
    params.model_version = modelVersion;
  }

  const response = await api.get("/api/predictions", {
    params,
  });

  return response.data;
};

/* ============================================================
   PREDICTION DETAIL
   ============================================================ */

export const getPredictionDetail = async (predictionId) => {
  const response = await api.get(`/api/predictions/${predictionId}`);

  return response.data;
};

/* ============================================================
   DASHBOARD ANALYTICS
   ============================================================ */

export const getDashboardSummary = async () => {
  const response = await api.get("/api/dashboard/summary");

  return response.data;
};

export const getROIDistribution = async () => {
  const response = await api.get("/api/dashboard/roi-distribution");

  return response.data;
};

export const getChannelPerformance = async () => {
  const response = await api.get("/api/dashboard/channel-performance");

  return response.data;
};

export const getCampaignTypePerformance = async () => {
  const response = await api.get("/api/dashboard/campaign-type-performance");

  return response.data;
};

export const getROITrend = async () => {
  const response = await api.get("/api/dashboard/roi-trend");

  return response.data;
};

export const getRecentPredictions = async (limit = 10) => {
  const response = await api.get("/api/dashboard/recent-predictions", {
    params: {
      limit,
    },
  });

  return response.data;
};

export default api;
