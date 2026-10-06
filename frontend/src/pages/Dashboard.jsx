/* eslint-disable react-hooks/set-state-in-effect */
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  getDashboardSummary,
  getROIDistribution,
  getChannelPerformance,
  getCampaignTypePerformance,
  getROITrend,
  getRecentPredictions,
  getModelMetadata,
} from "../services/api";

function formatNumber(value, maximumFractionDigits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return Number(value).toLocaleString("en-IN", {
    maximumFractionDigits,
  });
}

function formatROI(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return `${formatNumber(value, 2)}%`;
}

function getROICategory(roi) {
  const value = Number(roi);

  if (Number.isNaN(value)) {
    return "Uncategorized";
  }

  if (value < 0) {
    return "Negative";
  }

  if (value < 100) {
    return "Low";
  }

  if (value < 300) {
    return "Moderate";
  }

  if (value < 500) {
    return "High";
  }

  return "Very High";
}

function getCategoryClass(category) {
  switch (category) {
    case "Very High":
      return "roi-very-high";

    case "High":
      return "roi-high";

    case "Moderate":
      return "roi-moderate";

    case "Low":
      return "roi-low";

    case "Negative":
      return "roi-negative";

    default:
      return "roi-uncategorized";
  }
}

function Dashboard() {
  const navigate = useNavigate();

  const [summary, setSummary] = useState(null);
  const [roiDistribution, setROIDistribution] = useState([]);
  const [channelPerformance, setChannelPerformance] = useState([]);
  const [campaignTypePerformance, setCampaignTypePerformance] = useState([]);
  const [roiTrend, setROITrend] = useState([]);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [model, setModel] = useState(null);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const fetchDashboardData = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError("");

      const [
        summaryResponse,
        distributionResponse,
        channelResponse,
        campaignTypeResponse,
        trendResponse,
        recentResponse,
        modelResponse,
      ] = await Promise.all([
        getDashboardSummary(),
        getROIDistribution(),
        getChannelPerformance(),
        getCampaignTypePerformance(),
        getROITrend(),
        getRecentPredictions(10),
        getModelMetadata(),
      ]);

      if (!summaryResponse.success) {
        throw new Error("Unable to load dashboard summary.");
      }

      setSummary(summaryResponse.data);
      setROIDistribution(distributionResponse.data || []);
      setChannelPerformance(channelResponse.data || []);
      setCampaignTypePerformance(campaignTypeResponse.data || []);
      setROITrend(trendResponse.data || []);
      setRecentPredictions(recentResponse.data || []);

      if (modelResponse.success) {
        setModel(modelResponse.data);
      }
    } catch (err) {
      console.error("Dashboard loading error:", err);

      setError(
        err?.response?.data?.error?.message ||
          err?.message ||
          "Unable to load dashboard data.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);
  /*
   * ------------------------------------------------------------
   * PERFORMANCE HELPERS
   * ------------------------------------------------------------
   */

  const getPerformancePercentage = (value, collection) => {
    const numericValue = Number(value) || 0;

    const maximum = Math.max(
      ...collection.map((item) => Number(item.average_predicted_roi) || 0),
      0,
    );

    if (maximum <= 0) {
      return 0;
    }

    return Math.max(5, Math.min(100, (numericValue / maximum) * 100));
  };

  const sortByROI = (items) => {
    return [...items].sort(
      (a, b) =>
        Number(b.average_predicted_roi || 0) -
        Number(a.average_predicted_roi || 0),
    );
  };

  const sortedChannelPerformance = sortByROI(channelPerformance);

  const sortedCampaignTypePerformance = sortByROI(campaignTypePerformance);

  const latestTrend =
    roiTrend.length > 0 ? roiTrend[roiTrend.length - 1] : null;

  const previousTrend =
    roiTrend.length > 1 ? roiTrend[roiTrend.length - 2] : null;

  const latestTrendROI = latestTrend
    ? Number(latestTrend.average_predicted_roi) || 0
    : 0;

  const previousTrendROI = previousTrend
    ? Number(previousTrend.average_predicted_roi) || 0
    : 0;

  const trendChange =
    previousTrendROI !== 0
      ? ((latestTrendROI - previousTrendROI) / Math.abs(previousTrendROI)) * 100
      : null;

  /*
   * ------------------------------------------------------------
   * LOADING STATE
   * ------------------------------------------------------------
   */

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="dashboard-loading">
          <div className="loading-spinner"></div>
          <p>Loading campaign analytics...</p>
        </div>
      </div>
    );
  }

  /*
   * ------------------------------------------------------------
   * ERROR STATE
   * ------------------------------------------------------------
   */

  if (error && !summary) {
    return (
      <div className="dashboard-page">
        <div className="dashboard-error">
          <div className="error-icon">!</div>

          <div>
            <h2>Unable to load dashboard</h2>

            <p>{error}</p>

            <button
              type="button"
              className="dashboard-retry-button"
              onClick={() => fetchDashboardData()}
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-page">
      {/* ======================================================
          HEADER
      ====================================================== */}

      <section className="dashboard-header">
        <div>
          <p className="eyebrow">CAMPAIGN ANALYTICS</p>

          <h1>Campaign ROI Dashboard</h1>

          <p className="dashboard-subtitle">
            Monitor prediction activity, expected ROI, campaign performance, and
            model insights from one place.
          </p>
        </div>

        <div className="dashboard-header-actions">
          <div className="header-status">
            <span className="status-dot"></span>
            Model Active
          </div>

          <button
            type="button"
            className="refresh-button"
            onClick={() => fetchDashboardData(true)}
            disabled={refreshing}
          >
            <span
              className={refreshing ? "refresh-icon spinning" : "refresh-icon"}
            >
              ↻
            </span>

            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </section>

      {error && (
        <div className="dashboard-warning">
          <strong>Some dashboard data could not be refreshed.</strong>

          <span>{error}</span>
        </div>
      )}

      {/* ======================================================
          KPI CARDS
      ====================================================== */}

      <section className="dashboard-kpi-grid">
        <div className="analytics-kpi-card">
          <div className="analytics-kpi-top">
            <span className="analytics-kpi-label">TOTAL PREDICTIONS</span>

            <span className="analytics-kpi-icon">↗</span>
          </div>

          <strong className="analytics-kpi-value">
            {formatNumber(summary?.total_predictions, 0)}
          </strong>

          <span className="analytics-kpi-description">
            Predictions generated
          </span>
        </div>

        <div className="analytics-kpi-card">
          <div className="analytics-kpi-top">
            <span className="analytics-kpi-label">AVERAGE PREDICTED ROI</span>

            <span className="analytics-kpi-icon">%</span>
          </div>

          <strong className="analytics-kpi-value">
            {formatROI(summary?.average_predicted_roi)}
          </strong>

          <span className="analytics-kpi-description">
            Average expected return
          </span>
        </div>

        <div className="analytics-kpi-card">
          <div className="analytics-kpi-top">
            <span className="analytics-kpi-label">HIGHEST PREDICTED ROI</span>

            <span className="analytics-kpi-icon">★</span>
          </div>

          <strong className="analytics-kpi-value">
            {formatROI(summary?.highest_predicted_roi)}
          </strong>

          <span className="analytics-kpi-description">
            Highest expected return
          </span>
        </div>

        <div className="analytics-kpi-card">
          <div className="analytics-kpi-top">
            <span className="analytics-kpi-label">AVERAGE CAMPAIGN BUDGET</span>

            <span className="analytics-kpi-icon">₹</span>
          </div>

          <strong className="analytics-kpi-value">
            ₹{formatNumber(summary?.average_budget, 0)}
          </strong>

          <span className="analytics-kpi-description">
            Average planned budget
          </span>
        </div>

        <div className="analytics-kpi-card">
          <div className="analytics-kpi-top">
            <span className="analytics-kpi-label">TOTAL CAMPAIGNS</span>

            <span className="analytics-kpi-icon">◆</span>
          </div>

          <strong className="analytics-kpi-value">
            {formatNumber(summary?.total_campaigns, 0)}
          </strong>

          <span className="analytics-kpi-description">Campaign records</span>
        </div>
      </section>

      {/* ======================================================
          ROI TREND + ROI DISTRIBUTION
      ====================================================== */}

      <section className="dashboard-analytics-grid">
        {/* ROI TREND */}

        {/* ROI TREND */}

        <div className="dashboard-card dashboard-wide-card">
          <div className="card-header">
            <div>
              <p className="card-eyebrow">PERFORMANCE TREND</p>

              <h2>Predicted ROI Trend</h2>
            </div>

            <span className="chart-period">Historical predictions</span>
          </div>

          {roiTrend.length === 0 ? (
            <div className="empty-chart">
              <span>No trend data available.</span>
            </div>
          ) : (
            <>
              {/* TREND SUMMARY */}

              <div className="trend-summary">
                <div className="trend-summary-item">
                  <span>Latest Avg. ROI</span>

                  <strong>{formatROI(latestTrendROI)}</strong>
                </div>

                <div className="trend-summary-item">
                  <span>Latest Predictions</span>

                  <strong>
                    {formatNumber(latestTrend?.prediction_count, 0)}
                  </strong>
                </div>

                <div className="trend-summary-item">
                  <span>Change vs Previous</span>

                  <strong
                    className={
                      trendChange === null
                        ? ""
                        : trendChange >= 0
                          ? "trend-positive"
                          : "trend-negative"
                    }
                  >
                    {trendChange === null
                      ? "—"
                      : `${trendChange >= 0 ? "+" : ""}${formatNumber(
                          trendChange,
                          2,
                        )}%`}
                  </strong>
                </div>
              </div>

              {/* TREND CHART */}

              <div className="roi-trend-chart polished-trend-chart">
                <div className="trend-y-axis">
                  <span>
                    {formatROI(
                      Math.max(
                        ...roiTrend.map(
                          (item) => Number(item.average_predicted_roi) || 0,
                        ),
                      ),
                    )}
                  </span>

                  <span>75%</span>
                  <span>50%</span>
                  <span>25%</span>
                  <span>0%</span>
                </div>

                <div className="trend-chart-area">
                  <div className="trend-grid-lines">
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>

                  <div className="trend-bars polished-trend-bars">
                    {roiTrend.map((item, index) => {
                      const maxROI = Math.max(
                        ...roiTrend.map((trend) =>
                          Math.abs(Number(trend.average_predicted_roi) || 0),
                        ),
                        0,
                      );

                      const currentROI =
                        Number(item.average_predicted_roi) || 0;

                      const height =
                        maxROI > 0
                          ? Math.max(8, (Math.abs(currentROI) / maxROI) * 100)
                          : 8;

                      return (
                        <div
                          className="trend-column polished-trend-column"
                          key={`${item.date}-${index}`}
                          title={`${item.date}: ${formatROI(currentROI)} from ${
                            item.prediction_count
                          } prediction${
                            item.prediction_count !== 1 ? "s" : ""
                          }`}
                        >
                          <div className="trend-value">
                            {formatROI(currentROI)}
                          </div>

                          <div className="trend-bar-container polished-trend-bar-container">
                            <div
                              className="trend-bar polished-trend-bar"
                              style={{
                                height: `${height}%`,
                              }}
                            ></div>
                          </div>

                          <div className="trend-column-meta">
                            <span className="trend-date">
                              {new Date(
                                `${item.date}T00:00:00`,
                              ).toLocaleDateString("en-IN", {
                                day: "2-digit",
                                month: "short",
                              })}
                            </span>

                            <span className="trend-count">
                              {item.prediction_count} prediction
                              {item.prediction_count !== 1 ? "s" : ""}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* ROI DISTRIBUTION */}

        <div className="dashboard-card">
          <div className="card-header">
            <div>
              <p className="card-eyebrow">ROI ANALYSIS</p>

              <h2>ROI Distribution</h2>
            </div>
          </div>

          {roiDistribution.length === 0 ? (
            <div className="empty-chart">
              <span>No ROI distribution data.</span>
            </div>
          ) : (
            <div className="distribution-list">
              {roiDistribution.map((item) => (
                <div className="distribution-item" key={item.roi_category}>
                  <div className="distribution-info">
                    <span
                      className={`roi-badge ${getCategoryClass(
                        item.roi_category,
                      )}`}
                    >
                      {item.roi_category}
                    </span>

                    <strong>{item.prediction_count}</strong>
                  </div>

                  <div className="distribution-track">
                    <div
                      className="distribution-fill"
                      style={{
                        width: `${
                          summary?.total_predictions
                            ? Math.min(
                                100,
                                (item.prediction_count /
                                  summary.total_predictions) *
                                  100,
                              )
                            : 0
                        }%`,
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* ======================================================
          CHANNEL + CAMPAIGN TYPE
      ====================================================== */}

      <section className="dashboard-two-column-grid">
        {/* CHANNEL PERFORMANCE */}

        <div className="dashboard-card">
          <div className="card-header">
            <div>
              <p className="card-eyebrow">CHANNEL ANALYSIS</p>

              <h2>ROI by Marketing Channel</h2>
            </div>
          </div>

          {sortedChannelPerformance.length === 0 ? (
            <div className="empty-chart">
              <span>No channel data available.</span>
            </div>
          ) : (
            <div className="performance-list">
              {sortedChannelPerformance.map((item, index) => {
                const percentage = getPerformancePercentage(
                  item.average_predicted_roi,
                  sortedChannelPerformance,
                );

                return (
                  <div
                    className="performance-card-row"
                    key={item.marketing_channel}
                  >
                    <div className="performance-rank">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="performance-content">
                      <div className="performance-heading">
                        <strong>{item.marketing_channel}</strong>

                        <span>
                          {item.prediction_count} prediction
                          {item.prediction_count !== 1 ? "s" : ""}
                        </span>
                      </div>

                      <div className="performance-progress">
                        <div
                          className="performance-progress-fill"
                          style={{
                            width: `${percentage}%`,
                          }}
                        ></div>
                      </div>
                    </div>

                    <div className="performance-result">
                      <strong>{formatROI(item.average_predicted_roi)}</strong>

                      <span>Avg. ROI</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* CAMPAIGN TYPE PERFORMANCE */}

        <div className="dashboard-card">
          <div className="card-header">
            <div>
              <p className="card-eyebrow">CAMPAIGN ANALYSIS</p>

              <h2>ROI by Campaign Type</h2>
            </div>
          </div>

          {sortedCampaignTypePerformance.length === 0 ? (
            <div className="empty-chart">
              <span>No campaign type data available.</span>
            </div>
          ) : (
            <div className="performance-list">
              {sortedCampaignTypePerformance.map((item, index) => {
                const percentage = getPerformancePercentage(
                  item.average_predicted_roi,
                  sortedCampaignTypePerformance,
                );

                return (
                  <div
                    className="performance-card-row"
                    key={item.campaign_type}
                  >
                    <div className="performance-rank">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="performance-content">
                      <div className="performance-heading">
                        <strong>{item.campaign_type}</strong>

                        <span>
                          {item.prediction_count} prediction
                          {item.prediction_count !== 1 ? "s" : ""}
                        </span>
                      </div>

                      <div className="performance-progress">
                        <div
                          className="performance-progress-fill"
                          style={{
                            width: `${percentage}%`,
                          }}
                        ></div>
                      </div>
                    </div>

                    <div className="performance-result">
                      <strong>{formatROI(item.average_predicted_roi)}</strong>

                      <span>Avg. ROI</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </section>

      {/* ======================================================
    RECENT PREDICTIONS
====================================================== */}

      <section className="dashboard-card recent-predictions-card">
        <div className="card-header recent-predictions-header">
          <div>
            <p className="card-eyebrow">ACTIVITY</p>

            <h2>Recent Predictions</h2>

            <p className="section-description">
              Latest campaign ROI predictions generated by the system.
            </p>
          </div>

          <button
            type="button"
            className="view-all-button"
            onClick={() => navigate("/history")}
          >
            View All
            <span>→</span>
          </button>
        </div>

        {recentPredictions.length === 0 ? (
          <div className="empty-table">
            <div className="empty-table-icon">—</div>

            <h3>No predictions yet</h3>

            <p>Generate your first campaign ROI prediction to see it here.</p>

            <button
              type="button"
              onClick={() => navigate("/predict")}
              className="primary-dashboard-button"
            >
              Create Prediction
            </button>
          </div>
        ) : (
          <div className="dashboard-table-wrapper">
            <table className="dashboard-table polished-dashboard-table">
              <thead>
                <tr>
                  <th>Campaign</th>
                  <th>Type</th>
                  <th>Channel</th>
                  <th>Budget</th>
                  <th>Predicted ROI</th>
                  <th>Model</th>
                  <th>Created</th>
                </tr>
              </thead>

              <tbody>
                {recentPredictions.map((prediction) => {
                  const category =
                    prediction.roi_category ||
                    getROICategory(prediction.predicted_roi);

                  return (
                    <tr
                      key={prediction.prediction_id}
                      className="prediction-table-row"
                      onClick={() =>
                        navigate(`/history/${prediction.prediction_id}`, {
                          state: {
                            prediction,
                          },
                        })
                      }
                    >
                      {/* CAMPAIGN */}

                      <td className="prediction-campaign-cell">
                        <div className="campaign-table-name">
                          {prediction.campaign_name}
                        </div>
                      </td>

                      {/* TYPE */}

                      <td>
                        <span className="table-text-primary">
                          {prediction.campaign_type}
                        </span>
                      </td>

                      {/* CHANNEL */}

                      <td>
                        <span className="channel-pill">
                          {prediction.marketing_channel}
                        </span>
                      </td>

                      {/* BUDGET */}

                      <td>
                        <span className="budget-value">
                          ₹{formatNumber(prediction.budget, 0)}
                        </span>
                      </td>

                      {/* ROI */}

                      <td>
                        <div className="table-roi polished-table-roi">
                          <div className="roi-value-wrapper">
                            <strong className="table-roi-value">
                              {formatROI(prediction.predicted_roi)}
                            </strong>

                            <span className="roi-label">Predicted</span>
                          </div>

                          <span
                            className={`roi-badge ${getCategoryClass(
                              category,
                            )}`}
                          >
                            {category}
                          </span>
                        </div>
                      </td>

                      {/* MODEL */}

                      <td>
                        <span className="model-version-badge polished-model-badge">
                          v{prediction.model_version}
                        </span>
                      </td>

                      {/* CREATED */}

                      <td>
                        <div className="created-date">
                          <strong>
                            {new Date(prediction.created_at).toLocaleDateString(
                              "en-IN",
                              {
                                day: "2-digit",
                                month: "short",
                              },
                            )}
                          </strong>

                          <span>
                            {new Date(prediction.created_at).toLocaleTimeString(
                              "en-IN",
                              {
                                hour: "2-digit",
                                minute: "2-digit",
                              },
                            )}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ======================================================
          ACTIVE ML MODEL
      ====================================================== */}

      {model && (
        <section className="dashboard-model-section">
          <div className="dashboard-card">
            {/* MODEL HEADER */}

            <div className="model-section-header">
              <div className="model-heading">
                <div className="model-heading-icon">ML</div>

                <div>
                  <p className="card-eyebrow">CURRENT PRODUCTION MODEL</p>

                  <h2>Active ML Model</h2>

                  <p className="model-heading-subtitle">{model.model_name}</p>
                </div>
              </div>

              <div className="model-status-group">
                <span className="active-badge">
                  <span className="active-badge-dot"></span>
                  ACTIVE
                </span>

                <span className="model-version-large">v{model.version}</span>
              </div>
            </div>

            {/* MODEL CONTENT */}

            <div className="model-section-content">
              {/* MODEL DETAILS */}

              <div className="model-info-panel">
                <div className="model-panel-header">
                  <span className="model-panel-icon">◈</span>

                  <div>
                    <h3>Model Details</h3>

                    <p>Configuration and training information</p>
                  </div>
                </div>

                <div className="model-info-grid">
                  <div className="model-info-item">
                    <span>Algorithm</span>

                    <strong>{model.algorithm}</strong>
                  </div>

                  <div className="model-info-item">
                    <span>Version</span>

                    <strong>{model.version}</strong>
                  </div>

                  <div className="model-info-item">
                    <span>Features</span>

                    <strong>{model.feature_count}</strong>
                  </div>

                  <div className="model-info-item">
                    <span>Training Samples</span>

                    <strong>{formatNumber(model.training_samples, 0)}</strong>
                  </div>
                </div>

                <div className="model-trained">
                  <span>Last trained</span>

                  <strong>
                    {model.trained_at
                      ? new Date(model.trained_at).toLocaleDateString("en-IN", {
                          day: "2-digit",
                          month: "short",
                          year: "numeric",
                        })
                      : "—"}
                  </strong>
                </div>
              </div>

              {/* MODEL PERFORMANCE */}

              <div className="model-performance-panel">
                <div className="model-panel-header">
                  <span className="model-panel-icon">◎</span>

                  <div>
                    <h3>Model Performance</h3>

                    <p>Evaluation results from validation</p>
                  </div>
                </div>

                <div className="model-metrics-grid">
                  <div className="model-metric-card">
                    <span>Test R²</span>

                    <strong>{Number(model.test_r2).toFixed(3)}</strong>

                    <small>Test performance</small>
                  </div>

                  <div className="model-metric-card">
                    <span>CV R²</span>

                    <strong>{Number(model.cv_r2).toFixed(3)}</strong>

                    <small>Cross-validation</small>
                  </div>

                  <div className="model-metric-card">
                    <span>MAE</span>

                    <strong>{formatNumber(model.mae, 2)}</strong>

                    <small>Mean absolute error</small>
                  </div>

                  <div className="model-metric-card">
                    <span>RMSE</span>

                    <strong>{formatNumber(model.rmse, 2)}</strong>

                    <small>Root mean squared error</small>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
export default Dashboard;
