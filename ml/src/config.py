from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ML_DIR = PROJECT_ROOT / "ml"
MODEL_DIR = ML_DIR / "models"


# ============================================================
# DATA FILES
# ============================================================

RAW_DATA_FILE = (
    RAW_DATA_DIR /
    "Campaign_ROI_Estimation_Dataset_150000.xlsx"
)

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "campaign_roi_ml_dataset.csv"
)

SHEET_NAME = "Fact_CampaignPerformance"


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "ROI"


# ============================================================
# FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    "Budget",
    "CompetitorScore",
    "CampaignDurationDays",
    "DiscountPercent",
]

CATEGORICAL_FEATURES = [
    "Platform",
    "Region",
    "Device",
    "CustomerSegment",
    "ProductCategory",
    "CampaignType",
    "Season",
    "MarketingObjective",
]

DATE_DERIVED_FEATURES = [
    "CampaignYear",
    "CampaignMonth",
    "CampaignQuarter",
    "CampaignDayOfWeek",
]


FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + DATE_DERIVED_FEATURES
)


# ============================================================
# MODEL OUTPUT FILES
# ============================================================

MODEL_FILE = (
    MODEL_DIR /
    "campaign_roi_model.joblib"
)

METADATA_FILE = (
    MODEL_DIR /
    "model_metadata.json"
)

FEATURE_METADATA_FILE = (
    MODEL_DIR /
    "feature_metadata.json"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5