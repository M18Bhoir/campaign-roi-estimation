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
                    │        Vite         │
                    └──────────┬──────────┘
                               │
                               │ REST API
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Backend  │
                    │                     │
                    │ Prediction Service  │
                    │ Dashboard Service   │
                    │ Model Metadata      │
                    └───────┬───────┬─────┘
                            │       │
                ┌───────────┘       └────────────┐
                ▼                                ▼
       ┌─────────────────┐             ┌─────────────────┐
       │   PostgreSQL    │             │    ML Model     │
       │                 │             │ Random Forest   │
       │ Campaigns       │             │                 │
       │ Predictions     │             │ Joblib Artifact │
       │ Model Metadata  │             └─────────────────┘
       └─────────────────┘
```

---

## 🛠️ Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- Uvicorn
- PostgreSQL
- Psycopg

### Frontend

- React
- Vite
- JavaScript
- Axios
- Recharts

### Development & Deployment

- Git
- GitHub
- Git LFS
- Virtual Environment
- Environment Variables
- Docker-ready architecture

---

## 📁 Project Structure

```text
campaign-roi-estimation/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── create_tables.py
│   │   ├── logging_config.py
│   │   │
│   │   ├── repositories/
│   │   │   ├── campaign_repository.py
│   │   │   ├── prediction_repository.py
│   │   │   └── model_metadata_repository.py
│   │   │
│   │   └── services/
│   │       ├── prediction_service.py
│   │       ├── model_metadata_service.py
│   │       ├── dashboard_service.py
│   │       └── roi_category.py
│   │
│   ├── tests/
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── services/
│   │   └── ...
│   ├── package.json
│   └── .env.example
│
├── ml/
│   ├── notebooks/
│   ├── models/
│   │   ├── campaign_roi_model.joblib
│   │   ├── campaign_roi_model_transformed.joblib
│   │   ├── model_metadata.json
│   │   └── feature_metadata.json
│   └── results/
│
├── data/
│   ├── raw/
│   │   └── Campaign_ROI_Estimation_Dataset_150000.xlsx
│   └── processed/
│       └── campaign_roi_ml_dataset.csv
│
├── .gitignore
├── .gitattributes
├── .env.example
└── README.md
```

---

## 🗄️ Database

The backend uses PostgreSQL with SQLAlchemy.

### Main Tables

#### `campaigns`

Stores campaign information and campaign-related metrics.

#### `predictions`

Stores generated ROI predictions, prediction metadata, model version, and prediction latency.

#### `model_metadata`

Stores information about the active machine learning model and its evaluation metrics.

The prediction records preserve the model name and version used to generate each prediction.

---

## 🔌 API

The backend exposes REST endpoints for prediction, dashboard analytics, prediction history, and model information.

### Dashboard

```text
GET /api/dashboard/summary
GET /api/dashboard/roi-distribution
GET /api/dashboard/channel-performance
GET /api/dashboard/campaign-type-performance
GET /api/dashboard/roi-trend
GET /api/dashboard/recent-predictions
```

### Model

```text
GET /api/model/metadata
```

### Predictions

```text
GET /api/predictions/{prediction_id}
```

The development environment also exposes FastAPI Swagger/OpenAPI documentation.

---

## ⚙️ Environment Configuration

The project uses environment variables for configuration and secrets.

Create the required environment files from the provided examples.

### Root `.env`

```env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_DATABASE_PASSWORD@localhost:5432/campaign_roi_db
TEST_DATABASE_URL=postgresql+psycopg2://postgres:YOUR_DATABASE_PASSWORD@localhost:5432/campaign_roi_test_db
ENVIRONMENT=development
FRONTEND_URL=http://127.0.0.1:5173
LOG_LEVEL=INFO
```

### Frontend `.env`

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

> **Never commit real `.env` files or database credentials to GitHub.**

---

## ▶️ Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/M18Bhoir/campaign-roi-estimation.git
cd campaign-roi-estimation
```

### 2. Backend Setup

Create and activate the Python virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

Configure the backend environment variables using:

```text
backend/.env.example
```

Create the PostgreSQL database and initialize the application tables.

Start FastAPI:

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Development API documentation:

```text
http://127.0.0.1:8000/docs
```

### 3. Frontend Setup

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://127.0.0.1:5173
```

---

## 🔐 Security

The project includes environment-based security controls:

- Database credentials stored through environment variables
- `.env` excluded from Git
- Production CORS restriction
- Development-only Swagger/OpenAPI documentation
- Controlled API error responses
- Input validation through Pydantic
- Database validation rules
- Model version tracking for predictions

---

## 📦 Git LFS

The trained Random Forest model artifacts are stored using **Git Large File Storage (Git LFS)** because they are larger than GitHub's recommended normal Git file size.

Tracked LFS artifacts:

```text
ml/models/campaign_roi_model.joblib
ml/models/campaign_roi_model_transformed.joblib
```

After cloning the repository, ensure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

---

## 🧪 Testing

The backend includes automated tests covering API behavior, validation, prediction functionality, dashboard functionality, and database-related workflows.

The project test suite contains **88 passing tests**.

Run the test suite with:

```powershell
pytest
```

---

## 📈 Current Model Status

The current Random Forest model is marked as:

```text
Production Status: candidate
Model Version: 1.0.0
```

The model is designed for **pre-campaign ROI estimation**, allowing users to estimate expected campaign performance before relying on post-campaign outcome variables.

---

## 🔮 Future Improvements

Planned improvements include:

- Model performance improvement through additional feature engineering
- Hyperparameter optimization
- Model monitoring
- Prediction drift monitoring
- Automated model retraining
- Authentication and authorization
- Docker containerization
- CI/CD pipeline
- Cloud deployment
- Production database deployment
- Advanced campaign analytics

---

## 👨‍💻 Author

**Manas Kiran Bhoir**

BE Information Technology  
A. P. Shah Institute of Technology

GitHub: https://github.com/M18Bhoir

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.
