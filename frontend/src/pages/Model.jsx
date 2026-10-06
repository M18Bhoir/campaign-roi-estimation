/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from "react";
import "./model.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function Model() {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchModelMetadata = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${API_BASE_URL}/api/model/metadata`);

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result?.error?.message || "Failed to load model information.",
        );
      }

      setModel(result.data);
    } catch (err) {
      console.error("Model metadata error:", err);

      setError(err.message || "Unable to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelMetadata();
  }, []);

  const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined) {
      return "—";
    }

    return Number(value).toLocaleString("en-IN", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  const formatDate = (dateString) => {
    if (!dateString) {
      return "—";
    }

    return new Date(dateString).toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  /* ---------------- LOADING ---------------- */

  if (loading) {
    return (
      <div className="model-page">
        <main className="model-container">
          <div className="model-loading">
            <div className="skeleton skeleton-label"></div>

            <div className="skeleton skeleton-title"></div>

            <div className="skeleton skeleton-description"></div>

            <div className="model-kpi-grid">
              {[1, 2, 3, 4].map((item) => (
                <div key={item} className="model-skeleton-card"></div>
              ))}
            </div>

            <div className="model-skeleton-large"></div>
          </div>
        </main>
      </div>
    );
  }

  /* ---------------- ERROR ---------------- */

  if (error) {
    return (
      <div className="model-page">
        <main className="model-container">
          <div className="model-page-header">
            <p className="model-eyebrow">Campaign Analytics</p>

            <h1 className="model-page-title">Model Information</h1>

            <p className="model-page-description">
              View the currently active machine learning model and its
              evaluation metrics.
            </p>
          </div>

          <div className="model-error">
            <h2>Model Error</h2>

            <p>{error}</p>

            <button onClick={fetchModelMetadata} className="model-button">
              Try Again
            </button>
          </div>
        </main>
      </div>
    );
  }

  /* ---------------- MAIN PAGE ---------------- */

  return (
    <div className="model-page">
      <main className="model-container">
        {/* PAGE HEADER */}

        <div className="model-page-header">
          <div>
            <p className="model-eyebrow">Campaign Analytics</p>

            <h1 className="model-page-title">Model Information</h1>

            <p className="model-page-description">
              View the currently active machine learning model, evaluation
              metrics, and training information.
            </p>
          </div>

          <div className="model-status">
            <span
              className={`model-status-dot ${
                model?.is_active ? "active" : "inactive"
              }`}
            ></span>

            <span>{model?.is_active ? "Model Active" : "Inactive"}</span>
          </div>
        </div>

        {/* KPI CARDS */}

        <div className="model-kpi-grid">
          {/* TEST R2 */}

          <div className="model-card">
            <p className="model-card-label">Model Test R²</p>

            <p className="model-card-value">
              {formatNumber(model?.test_r2, 3)}
            </p>

            <p className="model-card-description">Test-set performance</p>
          </div>

          {/* CV R2 */}

          <div className="model-card">
            <p className="model-card-label">Cross-Validation R²</p>

            <p className="model-card-value">{formatNumber(model?.cv_r2, 3)}</p>

            <p className="model-card-description">Time-series validation</p>
          </div>

          {/* TRAINING SAMPLES */}

          <div className="model-card">
            <p className="model-card-label">Training Samples</p>

            <p className="model-card-value">
              {model?.training_samples?.toLocaleString("en-IN")}
            </p>

            <p className="model-card-description">Campaign records</p>
          </div>

          {/* FEATURES */}

          <div className="model-card">
            <p className="model-card-label">Model Features</p>

            <p className="model-card-value">{model?.feature_count}</p>

            <p className="model-card-description">Input features</p>
          </div>
        </div>

        {/* MODEL OVERVIEW + ERROR METRICS */}

        <div className="model-two-column">
          {/* MODEL OVERVIEW */}

          <section className="model-section">
            <div className="model-section-header">
              <div>
                <p className="model-section-label">Current Model</p>

                <h2 className="model-section-title">Model Overview</h2>
              </div>

              <span
                className={`model-active-badge ${
                  model?.is_active ? "active" : "inactive"
                }`}
              >
                {model?.is_active ? "Active" : "Inactive"}
              </span>
            </div>

            <div className="model-details">
              <div className="model-detail-row">
                <span>Model Name</span>

                <strong>{model?.model_name}</strong>
              </div>

              <div className="model-detail-row">
                <span>Algorithm</span>

                <strong>{model?.algorithm}</strong>
              </div>

              <div className="model-detail-row">
                <span>Version</span>

                <strong>{model?.version}</strong>
              </div>

              <div className="model-detail-row">
                <span>Features</span>

                <strong>{model?.feature_count}</strong>
              </div>

              <div className="model-detail-row">
                <span>Training Samples</span>

                <strong>
                  {model?.training_samples?.toLocaleString("en-IN")}
                </strong>
              </div>

              <div className="model-detail-row">
                <span>Trained At</span>

                <strong>{formatDate(model?.trained_at)}</strong>
              </div>
            </div>
          </section>

          {/* ERROR METRICS */}

          <section className="model-section">
            <div>
              <p className="model-section-label">Model Evaluation</p>

              <h2 className="model-section-title">Error Metrics</h2>
            </div>

            <div className="model-error-grid">
              {/* MAE */}

              <div className="model-metric-card">
                <p>MAE</p>

                <strong>{formatNumber(model?.mae)}</strong>

                <span>Mean Absolute Error</span>
              </div>

              {/* RMSE */}

              <div className="model-metric-card">
                <p>RMSE</p>

                <strong>{formatNumber(model?.rmse)}</strong>

                <span>Root Mean Squared Error</span>
              </div>
            </div>

            <div className="model-info-box">
              Lower MAE and RMSE indicate lower prediction error on the
              evaluation dataset.
            </div>
          </section>
        </div>

        {/* MODEL IDENTIFICATION */}

        <section className="model-section model-full-section">
          <div>
            <p className="model-section-label">Model Identification</p>

            <h2 className="model-section-title">Deployment Information</h2>
          </div>

          <div className="model-deployment-grid">
            {/* MODEL ID */}

            <div className="model-deployment-card">
              <p>Model ID</p>

              <strong className="model-id">{model?.model_id}</strong>
            </div>

            {/* VERSION */}

            <div className="model-deployment-card">
              <p>Version</p>

              <strong>{model?.version}</strong>
            </div>

            {/* STATUS */}

            <div className="model-deployment-card">
              <p>Status</p>

              <strong
                className={
                  model?.is_active ? "production-active" : "production-inactive"
                }
              >
                {model?.is_active ? "Production Active" : "Inactive"}
              </strong>
            </div>
          </div>
        </section>

        {/* EVALUATION SUMMARY */}

        <section className="model-section model-full-section">
          <p className="model-section-label">Evaluation Summary</p>

          <h2 className="model-section-title">Model Performance</h2>

          <div className="model-performance-grid">
            {/* TEST R2 */}

            <div className="performance-item">
              <div className="performance-header">
                <span>Test R²</span>

                <strong>{formatNumber(model?.test_r2, 3)}</strong>
              </div>

              <div className="performance-bar">
                <div
                  className="performance-fill"
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(100, (model?.test_r2 || 0) * 100),
                    )}%`,
                  }}
                ></div>
              </div>
            </div>

            {/* CROSS VALIDATION R2 */}

            <div className="performance-item">
              <div className="performance-header">
                <span>Cross-Validation R²</span>

                <strong>{formatNumber(model?.cv_r2, 3)}</strong>
              </div>

              <div className="performance-bar">
                <div
                  className="performance-fill"
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(100, (model?.cv_r2 || 0) * 100),
                    )}%`,
                  }}
                ></div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Model;
