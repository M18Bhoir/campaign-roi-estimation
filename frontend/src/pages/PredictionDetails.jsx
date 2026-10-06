/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from "react";

import { useLocation, useNavigate, useParams } from "react-router-dom";

import { getPredictionDetail } from "../services/api";

/* ============================================================
   FORMATTERS
   ============================================================ */

function formatCurrency(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return "—";
  }

  return `₹${number.toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  })}`;
}

function formatROI(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return "—";
  }

  return `${number.toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  })}%`;
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return "—";
  }

  return number.toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  });
}

function formatDate(value) {
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
}

/* ============================================================
   ROI HELPERS
   ============================================================ */

function getROICategory(roi) {
  const value = Number(roi);

  if (Number.isNaN(value)) {
    return "Not specified";
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

function getROIBadgeClass(roi) {
  const value = Number(roi);

  if (Number.isNaN(value)) {
    return "roi-badge roi-low";
  }

  if (value < 0) {
    return "roi-badge roi-negative";
  }

  if (value >= 1000) {
    return "roi-badge roi-high";
  }

  if (value >= 500) {
    return "roi-badge roi-medium";
  }

  return "roi-badge roi-low";
}

/* ============================================================
   PREDICTION DETAILS
   ============================================================ */

function PredictionDetails() {
  const navigate = useNavigate();

  const { predictionId } = useParams();

  const location = useLocation();

  /* ------------------------------------------------------------
     ROUTER STATE
     ------------------------------------------------------------ */

  const routerPrediction = location.state?.prediction;

  /* ------------------------------------------------------------
     STATE
     ------------------------------------------------------------ */

  const [prediction, setPrediction] = useState(routerPrediction || null);

  const [loading, setLoading] = useState(!routerPrediction);

  const [error, setError] = useState("");

  /* ------------------------------------------------------------
     LOAD PREDICTION
     ------------------------------------------------------------ */

  useEffect(() => {
    /*
     * If the user navigated from History/Dashboard,
     * React Router already supplied the prediction.
     */
    if (routerPrediction) {
      return;
    }

    /*
     * If there is no prediction ID in the URL,
     * we cannot call the backend.
     */
    if (!predictionId) {
      setError("Prediction ID is missing.");
      setLoading(false);
      return;
    }

    let cancelled = false;

    const loadPrediction = async () => {
      try {
        setLoading(true);
        setError("");

        const response = await getPredictionDetail(predictionId);

        if (!response || response.success !== true) {
          throw new Error("Unable to retrieve prediction details.");
        }

        if (!cancelled) {
          setPrediction(response.data);
        }
      } catch (err) {
        console.error("Prediction detail error:", err);

        if (!cancelled) {
          setPrediction(null);

          setError(
            err?.response?.data?.error?.message ||
              err?.message ||
              "Unable to load prediction details.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadPrediction();

    return () => {
      cancelled = true;
    };
  }, [predictionId, routerPrediction]);

  /* ------------------------------------------------------------
     ROI CATEGORY
     ------------------------------------------------------------ */

  const roiCategory =
    prediction?.roi_category || getROICategory(prediction?.predicted_roi);

  /* ============================================================
     LOADING STATE
     ============================================================ */

  if (loading) {
    return (
      <div className="prediction-details-page">
        <div className="prediction-details-loading">
          <div className="loading-spinner"></div>

          <p>Loading prediction details...</p>
        </div>
      </div>
    );
  }

  /* ============================================================
     ERROR STATE
     ============================================================ */

  if (error) {
    return (
      <div className="prediction-details-page">
        <div className="prediction-details-error">
          <div className="error-icon">!</div>

          <div>
            <h2>Unable to load prediction</h2>

            <p>{error}</p>

            <button type="button" onClick={() => navigate("/history")}>
              Back to History
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ============================================================
     FINAL SAFETY CHECK
     ============================================================ */

  if (!prediction) {
    return (
      <div className="prediction-details-page">
        <div className="prediction-details-error">
          <div>
            <h2>Prediction not found</h2>

            <p>The requested prediction could not be found.</p>

            <button type="button" onClick={() => navigate("/history")}>
              Back to History
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ============================================================
     MAIN PAGE
     ============================================================ */

  return (
    <div className="prediction-details-page">
      <div className="prediction-details-container">
        {/* ==================================================
            TOP BAR
            ================================================== */}

        <div className="details-topbar">
          <button
            type="button"
            className="back-button"
            onClick={() => navigate("/history")}
          >
            ← Back to History
          </button>

          <span className="details-status">SAVED PREDICTION</span>
        </div>

        {/* ==================================================
            HEADER
            ================================================== */}

        <div className="details-header">
          <div>
            <div className="section-label">CAMPAIGN PREDICTION</div>

            <h1>{prediction.campaign_name}</h1>

            <p>Complete details of the generated ROI prediction.</p>
          </div>

          <span className={getROIBadgeClass(prediction.predicted_roi)}>
            {formatROI(prediction.predicted_roi)}
          </span>
        </div>

        {/* ==================================================
            ROI SUMMARY
            ================================================== */}

        <div className="details-roi-card">
          <div className="details-roi-main">
            <span className="details-card-label">PREDICTED ROI</span>

            <strong>{formatROI(prediction.predicted_roi)}</strong>

            <p>
              Estimated return on investment generated by the active machine
              learning model.
            </p>
          </div>

          <div className="details-roi-side">
            <span className="details-card-label">ROI CATEGORY</span>

            <strong>{roiCategory}</strong>
          </div>
        </div>

        {/* ==================================================
            CAMPAIGN INFORMATION
            ================================================== */}

        <section className="details-section">
          <div className="details-section-header">
            <div className="section-label">CAMPAIGN INFORMATION</div>

            <h2>Campaign Details</h2>
          </div>

          <div className="details-grid">
            <div className="details-item">
              <span>Campaign Name</span>

              <strong>{prediction.campaign_name || "—"}</strong>
            </div>

            <div className="details-item">
              <span>Campaign Type</span>

              <strong>{prediction.campaign_type || "—"}</strong>
            </div>

            <div className="details-item">
              <span>Marketing Channel</span>

              <strong>{prediction.marketing_channel || "—"}</strong>
            </div>

            <div className="details-item">
              <span>Target Audience</span>

              <strong>{prediction.target_audience || "—"}</strong>
            </div>

            <div className="details-item">
              <span>Budget</span>

              <strong>{formatCurrency(prediction.budget)}</strong>
            </div>

            <div className="details-item">
              <span>Duration</span>

              <strong>
                {prediction.duration_days !== null &&
                prediction.duration_days !== undefined
                  ? `${prediction.duration_days} days`
                  : "—"}
              </strong>
            </div>
          </div>
        </section>

        {/* ==================================================
            MODEL OUTPUT
            ================================================== */}

        <section className="details-section">
          <div className="details-section-header">
            <div className="section-label">MODEL OUTPUT</div>

            <h2>Prediction Information</h2>
          </div>

          <div className="details-grid">
            <div className="details-item highlight-item">
              <span>Predicted ROI</span>

              <strong>{formatROI(prediction.predicted_roi)}</strong>
            </div>

            <div className="details-item">
              <span>Predicted Revenue</span>

              <strong>{formatCurrency(prediction.predicted_revenue)}</strong>
            </div>

            <div className="details-item">
              <span>Predicted Profit</span>

              <strong>{formatCurrency(prediction.predicted_profit)}</strong>
            </div>

            <div className="details-item">
              <span>ROI Category</span>

              <strong>{roiCategory}</strong>
            </div>
          </div>
        </section>

        {/* ==================================================
            MACHINE LEARNING
            ================================================== */}

        <section className="details-section">
          <div className="details-section-header">
            <div className="section-label">MACHINE LEARNING</div>

            <h2>Model Information</h2>
          </div>

          <div className="details-grid">
            <div className="details-item">
              <span>Model</span>

              <strong>{prediction.model_name || "—"}</strong>
            </div>

            <div className="details-item">
              <span>Version</span>

              <strong>
                {prediction.model_version
                  ? `v${prediction.model_version}`
                  : "—"}
              </strong>
            </div>

            <div className="details-item">
              <span>Prediction Latency</span>

              <strong>
                {prediction.prediction_latency_ms !== null &&
                prediction.prediction_latency_ms !== undefined
                  ? `${formatNumber(prediction.prediction_latency_ms)} ms`
                  : "—"}
              </strong>
            </div>

            <div className="details-item">
              <span>Prediction Created</span>

              <strong>{formatDate(prediction.created_at)}</strong>
            </div>
          </div>
        </section>

        {/* ==================================================
            RECORD INFORMATION
            ================================================== */}

        <section className="details-section">
          <div className="details-section-header">
            <div className="section-label">RECORD INFORMATION</div>

            <h2>Prediction Identifiers</h2>
          </div>

          <div className="details-id-grid">
            <div className="details-id-item">
              <span>Prediction ID</span>

              <code>{prediction.prediction_id || "—"}</code>
            </div>

            <div className="details-id-item">
              <span>Campaign ID</span>

              <code>{prediction.campaign_id || "—"}</code>
            </div>
          </div>
        </section>

        {/* ==================================================
            ACTIONS
            ================================================== */}

        <div className="details-actions">
          <button
            type="button"
            className="secondary-button"
            onClick={() => navigate("/history")}
          >
            ← Back to History
          </button>

          <button
            type="button"
            className="primary-button"
            onClick={() => navigate("/predict")}
          >
            Create New Prediction
          </button>
        </div>
      </div>
    </div>
  );
}

export default PredictionDetails;
