@"
# Campaign ROI Estimation

An end-to-end full-stack machine learning application for estimating the potential Return on Investment (ROI) of marketing campaigns.

The system combines a **Random Forest machine learning model**, **FastAPI backend**, **PostgreSQL database**, and **React frontend** to provide campaign ROI predictions, prediction history, campaign analytics, and model information.

---

## 🚀 Key Features

### Machine Learning
- Pre-campaign ROI prediction
- Random Forest Regression model
- 16 pre-campaign features
- Chronological train/test validation
- TimeSeriesSplit cross-validation
- Model versioning and metadata
- Prediction categorization based on estimated ROI

### Backend
- FastAPI REST API
- Pydantic request/response validation
- PostgreSQL database
- SQLAlchemy ORM
- Prediction history
- Campaign management
- Dashboard analytics
- Model metadata API
- Structured application logging
- Environment-based configuration
- Production-aware CORS configuration
- Swagger/OpenAPI documentation in development

### Frontend
- React + Vite
- Campaign ROI prediction interface
- Prediction details
- Prediction history
- Dashboard analytics
- ROI distribution
- Channel performance
- Campaign-type performance
- ROI trends
- Model information

---

## 🧠 Machine Learning

The application predicts campaign ROI using information available **before a campaign is executed**.

### Model

**Algorithm:** Random Forest Regressor

| Parameter | Value |
|---|---:|
| Estimators | 400 |
| Max Depth | 12 |
| Min Samples Split | 10 |
| Min Samples Leaf | 5 |
| Max Features | sqrt |
| Random State | 42 |

### Input Features

The production prediction pipeline uses 16 pre-campaign features.

#### Numerical Features
- Budget
- Competitor Score
- Campaign Duration Days
- Discount Percent

#### Categorical Features
- Platform
- Region
- Device
- Customer Segment
- Product Category
- Campaign Type
- Season
- Marketing Objective

#### Date-Derived Features
- Campaign Year
- Campaign Month
- Campaign Quarter
- Campaign Day of Week

Post-campaign and leakage-prone variables such as revenue, conversions, clicks, impressions, spend, CTR, and ROI are excluded from the prediction features.

---

## 📊 Model Performance

The final model was evaluated using a chronological 80/20 train-test split.

| Metric | Result |
|---|---:|
| Training Samples | 120,000 |
| Testing Samples | 30,000 |
| Features | 16 |
| Train R² | 0.409954 |
| Test R² | 0.302672 |
| Cross-Validation R² | 0.287157 |
| CV R² Std | 0.010263 |
| MAE | 2313.31 |
| RMSE | 6240.27 |
| Train-Test R² Gap | 0.107282 |

### Validation Strategy

- Chronological 80/20 train-test split
- TimeSeriesSplit cross-validation
- 5 cross-validation splits
- Random state: 42

The model is currently treated as a **candidate production model**.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │    React Frontend   │
                    │      Vite           │
                    └──────────┬──────────┘
                               │
                               │ REST API
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Backend   │
                    │                     │
                    │ Prediction Service  │
                    │ Dashboard Service   │
                    │ Model Metadata      │
                    └───────┬───────┬─────┘
                            │       │
                ┌───────────┘       └────────────┐
                ▼                                ▼
       ┌─────────────────┐             ┌─────────────────┐
       │   PostgreSQL    │             │  ML Model       │
       │                 │             │ Random Forest   │
       │ Campaigns       │             │                 │
       │ Predictions     │             │ Joblib Artifact │
       │ Model Metadata  │             └─────────────────┘
       └─────────────────┘
