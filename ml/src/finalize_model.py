import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

sys.path.append(str(PROJECT_ROOT / "ml" / "src"))


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

from config import (
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS,
)

from preprocessing import create_preprocessor


# ============================================================
# OUTPUT PATHS
# ============================================================

FINAL_MODEL_FILE = MODEL_DIR / "campaign_roi_model.joblib"
FINAL_METADATA_FILE = MODEL_DIR / "model_metadata.json"
FINAL_FEATURE_METADATA_FILE = MODEL_DIR / "feature_metadata.json"

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

FINAL_RESULTS_FILE = RESULTS_DIR / "final_model_results.json"
FINAL_TEST_PREDICTIONS_FILE = RESULTS_DIR / "final_test_predictions.csv"


# ============================================================
# DISPLAY HELPER
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DIRECTORY VALIDATION
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print_section("PHASE 3F.1 - FINAL MODEL TRAINING")

print_section("LOADING PROCESSED DATASET")

if not PROCESSED_DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{PROCESSED_DATA_FILE}"
    )

df = pd.read_csv(PROCESSED_DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

print_section("VALIDATING DATASET")

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )

print("All required columns are present.")


# ============================================================
# REMOVE INVALID TARGET VALUES
# ============================================================

initial_rows = len(df)

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[TARGET_COLUMN]
).reset_index(drop=True)

removed_rows = initial_rows - len(df)

print(f"Initial rows: {initial_rows:,}")
print(f"Rows removed: {removed_rows:,}")
print(f"Final rows  : {len(df):,}")


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


print_section("FEATURE CONFIGURATION")

print(f"Target column : {TARGET_COLUMN}")
print(f"Feature count : {len(FEATURE_COLUMNS)}")

print("\nFeatures:")

for index, feature in enumerate(FEATURE_COLUMNS, start=1):
    print(f"{index:2d}. {feature}")


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print_section("CHRONOLOGICAL DATA SPLIT")

split_index = int(
    len(df) * (1 - TEST_SIZE)
)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()


print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")

print(
    f"Training target mean: "
    f"{y_train.mean():,.4f}"
)

print(
    f"Testing target mean : "
    f"{y_test.mean():,.4f}"
)


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

print_section("CREATING PREPROCESSING PIPELINE")

preprocessor = create_preprocessor()

print("Preprocessor created successfully.")


# ============================================================
# FINAL RANDOM FOREST
# ============================================================

print_section("CREATING FINAL RANDOM FOREST")

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Random Forest configuration:")
print("  n_estimators      :", 400)
print("  max_depth         :", 12)
print("  min_samples_split :", 10)
print("  min_samples_leaf  :", 5)
print("  max_features      :", "sqrt")
print("  random_state      :", RANDOM_STATE)


# ============================================================
# COMPLETE PRODUCTION PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

print("\nProduction pipeline created:")
print("  preprocessing -> Random Forest")


# ============================================================
# TIME SERIES CROSS VALIDATION
# ============================================================

print_section("TIME SERIES CROSS VALIDATION")

tscv = TimeSeriesSplit(
    n_splits=N_SPLITS
)

cv_start_time = time.time()

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=tscv,
    scoring="r2",
    n_jobs=1,
)

cv_time = time.time() - cv_start_time


print("CV R² scores:")

for index, score in enumerate(cv_scores, start=1):
    print(f"Fold {index}: {score:.4f}")

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"\nMean CV R²: {cv_mean:.4f}")
print(f"CV R² Std : {cv_std:.4f}")
print(f"CV Time   : {cv_time:.2f} seconds")


# ============================================================
# FINAL TRAINING
# ============================================================

print_section("TRAINING FINAL MODEL")

train_start_time = time.time()

pipeline.fit(
    X_train,
    y_train
)

training_time = time.time() - train_start_time

print(
    f"Training completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# TRAIN PREDICTIONS
# ============================================================

print_section("CALCULATING TRAIN METRICS")

train_predictions = pipeline.predict(
    X_train
)

train_r2 = r2_score(
    y_train,
    train_predictions
)

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_predictions
    )
)


print(f"Train R²   : {train_r2:.4f}")
print(f"Train MAE  : {train_mae:.4f}")
print(f"Train RMSE : {train_rmse:.4f}")


# ============================================================
# TEST PREDICTIONS
# ============================================================

print_section("CALCULATING TEST METRICS")

test_predictions = pipeline.predict(
    X_test
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)


print(f"Test R²   : {test_r2:.4f}")
print(f"Test MAE  : {test_mae:.4f}")
print(f"Test RMSE : {test_rmse:.4f}")


# ============================================================
# TRAIN / TEST GAP
# ============================================================

r2_gap = train_r2 - test_r2

print_section("GENERALIZATION CHECK")

print(f"Train R²      : {train_r2:.4f}")
print(f"Test R²       : {test_r2:.4f}")
print(f"Train-Test Gap: {r2_gap:.4f}")


# ============================================================
# TEST PREDICTION DATASET
# ============================================================

print_section("CREATING FINAL TEST PREDICTIONS")

test_predictions_df = pd.DataFrame(
    {
        "Actual_ROI": y_test.values,
        "Predicted_ROI": test_predictions,
    }
)

test_predictions_df["Absolute_Error"] = np.abs(
    test_predictions_df["Actual_ROI"]
    - test_predictions_df["Predicted_ROI"]
)

test_predictions_df["Signed_Error"] = (
    test_predictions_df["Actual_ROI"]
    - test_predictions_df["Predicted_ROI"]
)

test_predictions_df["Absolute_Percentage_Error"] = np.where(
    test_predictions_df["Actual_ROI"] != 0,
    (
        test_predictions_df["Absolute_Error"]
        / np.abs(test_predictions_df["Actual_ROI"])
    ) * 100,
    np.nan,
)


test_predictions_df.to_csv(
    FINAL_TEST_PREDICTIONS_FILE,
    index=False
)

print(
    f"Test predictions saved:\n"
    f"{FINAL_TEST_PREDICTIONS_FILE}"
)


# ============================================================
# MODEL VERSION
# ============================================================

MODEL_VERSION = "1.0.0"


# ============================================================
# MODEL METADATA
# ============================================================

model_metadata = {
    "model_name": "Campaign ROI Random Forest",
    "model_version": MODEL_VERSION,
    "algorithm": "RandomForestRegressor",

    "target_column": TARGET_COLUMN,

    "test_r2": round(float(test_r2), 6),
    "cv_r2": round(float(cv_mean), 6),
    "cv_r2_std": round(float(cv_std), 6),

    "train_r2": round(float(train_r2), 6),

    "mae": round(float(test_mae), 6),
    "rmse": round(float(test_rmse), 6),

    "train_test_r2_gap": round(
        float(r2_gap),
        6
    ),

    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),

    "feature_count": int(len(FEATURE_COLUMNS)),

    "training_time_seconds": round(
        float(training_time),
        4
    ),

    "random_state": RANDOM_STATE,

    "hyperparameters": {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },

    "target_transformation": None,

    "validation_strategy": {
        "train_test_split": "chronological_80_20",
        "cross_validation": "TimeSeriesSplit",
        "n_splits": N_SPLITS,
    },

    "production_status": "candidate",
}


# ============================================================
# FEATURE METADATA
# ============================================================

numerical_features = [
    "Budget",
    "CompetitorScore",
    "CampaignDurationDays",
    "DiscountPercent",
]

categorical_features = [
    "Platform",
    "Region",
    "Device",
    "CustomerSegment",
    "ProductCategory",
    "CampaignType",
    "Season",
    "MarketingObjective",
]

date_derived_features = [
    "CampaignYear",
    "CampaignMonth",
    "CampaignQuarter",
    "CampaignDayOfWeek",
]


feature_metadata = {
    "model_name": "Campaign ROI Random Forest",
    "model_version": MODEL_VERSION,

    "target": TARGET_COLUMN,

    "feature_count": len(FEATURE_COLUMNS),

    "features": FEATURE_COLUMNS,

    "numerical_features": numerical_features,

    "categorical_features": categorical_features,

    "date_derived_features": date_derived_features,

    "feature_order": FEATURE_COLUMNS,

    "excluded_features": [
        "index",
        "CampaignID",
        "Spend",
        "Impressions",
        "Clicks",
        "CTR",
        "CPC",
        "CPM",
        "Leads",
        "Conversions",
        "ConversionRate",
        "Revenue",
        "Profit",
        "ROI",
        "Likes",
        "Shares",
        "Comments",
        "EngagementRate",
        "BounceRate",
        "SessionDuration",
        "CustomerRating",
        "CampaignStatus",
        "RepeatCustomers",
        "NewCustomers",
        "Orders",
        "AverageOrderValue",
        "RefundRate",
        "CampaignSuccess",
    ],

    "data_policy": {
        "leakage_prevention": True,
        "post_campaign_features_excluded": True,
        "target_transformation": False,
    },
}


# ============================================================
# SAVE METADATA
# ============================================================

print_section("SAVING MODEL METADATA")

with open(
    FINAL_METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_metadata,
        file,
        indent=4
    )


with open(
    FINAL_FEATURE_METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        feature_metadata,
        file,
        indent=4
    )


print(
    f"Model metadata saved:\n"
    f"{FINAL_METADATA_FILE}"
)

print(
    f"Feature metadata saved:\n"
    f"{FINAL_FEATURE_METADATA_FILE}"
)


# ============================================================
# SAVE MODEL
# ============================================================

print_section("SAVING FINAL MODEL PIPELINE")

joblib.dump(
    pipeline,
    FINAL_MODEL_FILE
)

print(
    f"Final model saved:\n"
    f"{FINAL_MODEL_FILE}"
)


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print_section("VERIFYING SAVED MODEL")

if not FINAL_MODEL_FILE.exists():
    raise FileNotFoundError(
        "Final model file was not created."
    )

if not FINAL_METADATA_FILE.exists():
    raise FileNotFoundError(
        "Model metadata file was not created."
    )

if not FINAL_FEATURE_METADATA_FILE.exists():
    raise FileNotFoundError(
        "Feature metadata file was not created."
    )


model_size_mb = (
    FINAL_MODEL_FILE.stat().st_size
    / (1024 * 1024)
)

print(
    f"Model file size: "
    f"{model_size_mb:.2f} MB"
)

print("Model file exists.")
print("Model metadata exists.")
print("Feature metadata exists.")


# ============================================================
# FINAL PREDICTION SANITY CHECK
# ============================================================

print_section("FINAL PREDICTION SANITY CHECK")

sample_input = X_test.iloc[
    [0]
].copy()

sample_prediction = pipeline.predict(
    sample_input
)

print("Sample input:")
print(sample_input.to_string(index=False))

print(
    f"\nSample predicted ROI: "
    f"{sample_prediction[0]:,.4f}"
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

final_results = {
    "model_name": "Campaign ROI Random Forest",
    "model_version": MODEL_VERSION,

    "dataset": {
        "rows": int(len(df)),
        "features": int(len(FEATURE_COLUMNS)),
        "target": TARGET_COLUMN,
    },

    "metrics": {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "cv_r2": float(cv_mean),
        "cv_r2_std": float(cv_std),
        "mae": float(test_mae),
        "rmse": float(test_rmse),
        "train_test_r2_gap": float(r2_gap),
    },

    "training": {
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "training_time_seconds": float(training_time),
    },

    "model_file": str(FINAL_MODEL_FILE),
    "metadata_file": str(FINAL_METADATA_FILE),
    "feature_metadata_file": str(
        FINAL_FEATURE_METADATA_FILE
    ),

    "status": "candidate",
}


with open(
    FINAL_RESULTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_results,
        file,
        indent=4
    )


# ============================================================
# COMPLETION
# ============================================================

print_section("PHASE 3F.1 COMPLETED")

print("Final model training completed successfully.")

print("\nFinal Metrics")
print("-" * 40)
print(f"CV R²       : {cv_mean:.4f}")
print(f"Test R²     : {test_r2:.4f}")
print(f"Test MAE    : {test_mae:.4f}")
print(f"Test RMSE   : {test_rmse:.4f}")
print(f"Train R²    : {train_r2:.4f}")
print(f"R² Gap      : {r2_gap:.4f}")

print("\nProduction Artifacts")
print("-" * 40)
print(f"Model       : {FINAL_MODEL_FILE}")
print(f"Metadata    : {FINAL_METADATA_FILE}")
print(f"Features    : {FINAL_FEATURE_METADATA_FILE}")
print(f"Results     : {FINAL_RESULTS_FILE}")
print(f"Predictions : {FINAL_TEST_PREDICTIONS_FILE}")

print("\nModel status: CANDIDATE")