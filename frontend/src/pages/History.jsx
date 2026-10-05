import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { getPredictionHistory } from "../services/api";

function History() {
  const navigate = useNavigate();

  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [total, setTotal] = useState(0);

  const [limit] = useState(20);
  const [offset, setOffset] = useState(0);

  /* ============================================================
     LOAD HISTORY
     ============================================================ */

  const loadHistory = useCallback(async () => {
    try {
      setLoading(true);
      setError("");

      const response = await getPredictionHistory({
        limit,
        offset,
      });

      if (!response || response.success !== true) {
        throw new Error("Unable to retrieve prediction history.");
      }

      setHistory(Array.isArray(response.data) ? response.data : []);
      setTotal(Number(response.total || 0));
    } catch (error) {
      console.error("Prediction history error:", error);

      setHistory([]);

      setError(
        error?.response?.data?.error?.message ||
          error?.message ||
          "Unable to load prediction history.",
      );
    } finally {
      setLoading(false);
    }
  }, [limit, offset]);

  /* ============================================================
     INITIAL LOAD
     ============================================================ */

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  /* ============================================================
     REFRESH
     ============================================================ */

  const handleRefresh = () => {
    loadHistory();
  };

  /* ============================================================
     OPEN PREDICTION
     ============================================================ */

  const handleViewPrediction = (prediction) => {
    if (!prediction?.prediction_id) {
      return;
    }

    navigate(`/history/${prediction.prediction_id}`, {
      state: {
        prediction,
      },
    });
  };

  /* ============================================================
     PAGINATION
     ============================================================ */

  const handlePrevious = () => {
    if (offset === 0) {
      return;
    }

    setOffset(Math.max(0, offset - limit));
  };

  const handleNext = () => {
    if (offset + limit >= total) {
      return;
    }

    setOffset(offset + limit);
  };

  const currentPage = total === 0 ? 1 : Math.floor(offset / limit) + 1;

  const totalPages = total === 0 ? 1 : Math.ceil(total / limit);

  /* ============================================================
     FORMATTERS
     ============================================================ */

  const formatROI = (value) => {
    if (value === null || value === undefined) {
      return "—";
    }

    const number = Number(value);

    if (Number.isNaN(number)) {
      return "—";
    }

    return `${number.toLocaleString("en-IN", {
      maximumFractionDigits: 2,
    })}%`;
  };

  const formatCurrency = (value) => {
    if (value === null || value === undefined) {
      return "—";
    }

    const number = Number(value);

    if (Number.isNaN(number)) {
      return "—";
    }

    return `₹${number.toLocaleString("en-IN", {
      maximumFractionDigits: 2,
    })}`;
  };

  const formatDate = (value) => {
    if (!value) {
      return "—";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
      return "—";
    }

    return date.toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getROIBadgeClass = (roi) => {
    const value = Number(roi);

    if (Number.isNaN(value)) {
      return "roi-badge roi-low";
    }

    if (value >= 1000) {
      return "roi-badge roi-high";
    }

    if (value >= 500) {
      return "roi-badge roi-medium";
    }

    if (value >= 0) {
      return "roi-badge roi-low";
    }

    return "roi-badge roi-negative";
  };

  /* ============================================================
     KEYBOARD ACCESS
     ============================================================ */

  const handleRowKeyDown = (event, prediction) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();

      handleViewPrediction(prediction);
    }
  };

  /* ============================================================
     RENDER
     ============================================================ */

  return (
    <div className="history-page">
      <div className="history-container">
        {/* ==================================================
            HEADER
            ================================================== */}

        <div className="history-header">
          <div>
            <div className="section-label">CAMPAIGN ANALYTICS</div>

            <h1>Prediction History</h1>

            <p>View previously generated campaign ROI predictions.</p>
          </div>

          <button
            type="button"
            className="refresh-button"
            onClick={handleRefresh}
            disabled={loading}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        {/* ==================================================
            ERROR
            ================================================== */}

        {error && (
          <div className="history-error">
            <div>
              <strong>History Error</strong>

              <p>{error}</p>
            </div>

            <button type="button" onClick={handleRefresh}>
              Try Again
            </button>
          </div>
        )}

        {/* ==================================================
            LOADING
            ================================================== */}

        {loading && !error && (
          <div className="history-card loading-card">
            <div className="loading-spinner" />

            <p>Loading prediction history...</p>
          </div>
        )}

        {/* ==================================================
            EMPTY STATE
            ================================================== */}

        {!loading && !error && history.length === 0 && (
          <div className="history-card empty-card">
            <div className="empty-icon">—</div>

            <h2>No Predictions Yet</h2>

            <p>Generate your first campaign ROI prediction to see it here.</p>
          </div>
        )}

        {/* ==================================================
            HISTORY TABLE
            ================================================== */}

        {!loading && !error && history.length > 0 && (
          <div className="history-card">
            <div className="table-header">
              <div>
                <h2>Campaign Predictions</h2>

                <p>
                  {total} prediction
                  {total !== 1 ? "s" : ""} available
                </p>
              </div>
            </div>

            <div className="table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Campaign</th>
                    <th>Type</th>
                    <th>Channel</th>
                    <th>Budget</th>
                    <th>Duration</th>
                    <th>Predicted ROI</th>
                    <th>Model</th>
                    <th>Created</th>
                  </tr>
                </thead>

                <tbody>
                  {history.map((prediction) => (
                    <tr
                      key={prediction.prediction_id}
                      className="clickable-history-row"
                      onClick={() => handleViewPrediction(prediction)}
                      onKeyDown={(event) => handleRowKeyDown(event, prediction)}
                      tabIndex={0}
                      role="link"
                      aria-label={`View prediction for ${
                        prediction.campaign_name
                      }`}
                    >
                      <td>
                        <div className="campaign-name">
                          {prediction.campaign_name}
                        </div>

                        <div className="campaign-id">
                          ID: {prediction.prediction_id?.slice(0, 8)}
                        </div>
                      </td>

                      <td>{prediction.campaign_type || "—"}</td>

                      <td>{prediction.marketing_channel || "—"}</td>

                      <td>{formatCurrency(prediction.budget)}</td>

                      <td>
                        {prediction.duration_days !== null &&
                        prediction.duration_days !== undefined
                          ? `${prediction.duration_days} days`
                          : "—"}
                      </td>

                      <td>
                        <span
                          className={getROIBadgeClass(prediction.predicted_roi)}
                        >
                          {formatROI(prediction.predicted_roi)}
                        </span>
                      </td>

                      <td>
                        <div className="model-name">
                          {prediction.model_name || "—"}
                        </div>

                        <div className="model-version">
                          {prediction.model_version
                            ? `v${prediction.model_version}`
                            : ""}
                        </div>
                      </td>

                      <td>{formatDate(prediction.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* ==================================================
                PAGINATION
                ================================================== */}

            <div className="pagination">
              <div className="pagination-info">
                Showing {offset + 1}–{Math.min(offset + history.length, total)}{" "}
                of {total}
              </div>

              <div className="pagination-controls">
                <button
                  type="button"
                  onClick={handlePrevious}
                  disabled={offset === 0}
                >
                  Previous
                </button>

                <span>
                  Page {currentPage} of {totalPages}
                </span>

                <button
                  type="button"
                  onClick={handleNext}
                  disabled={offset + limit >= total}
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default History;
