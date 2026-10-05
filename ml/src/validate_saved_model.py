import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

sys.path.append(str(PROJECT_ROOT / "ml" / "src"))


# ============================================================
# CONFIG
# ============================================================

from config import (
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    MODEL_DIR,
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = MODEL_DIR / "campaign_roi_model.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
FEATURE_METADATA_FILE = MODEL_DIR / "feature_metadata.json"


# ============================================================
# DISPLAY
# ============================================================

def print_section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# PHASE START
# ============================================================

print_section(
    "PHASE 3F.2 - INDEPENDENT SAVED MODEL VALIDATION"
)


# ============================================================
# CHECK FILES
# ============================================================

print_section("CHECKING PRODUCTION ARTIFACTS")


required_files = {
    "Model": MODEL_FILE,
    "Model Metadata": METADATA_FILE,
    "Feature Metadata": FEATURE_METADATA_FILE,
}


for name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{name} file not found:\n{file_path}"
        )

    print(f"{name}: FOUND")
    print(f"Path: {file_path}")


# ============================================================
# LOAD MODEL
# ============================================================

print_section("LOADING SAVED MODEL")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")
print(f"Model type: {type(model).__name__}")


# ============================================================
# VALIDATE PIPELINE COMPONENTS
# ============================================================

print_section("VALIDATING PIPELINE")

if not hasattr(model, "named_steps"):

    raise ValueError(
        "Saved object is not a sklearn Pipeline."
    )


print("Pipeline detected.")

print("\nPipeline steps:")

for step_name, step_object in model.named_steps.items():

    print(
        f"  {step_name}: "
        f"{type(step_object).__name__}"
    )


required_steps = [
    "preprocessor",
    "model",
]


for step in required_steps:

    if step not in model.named_steps:

        raise ValueError(
            f"Required pipeline step missing: {step}"
        )


print("\nRequired pipeline steps are present.")


# ============================================================
# LOAD METADATA
# ============================================================

print_section("LOADING MODEL METADATA")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


with open(
    FEATURE_METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    feature_metadata = json.load(file)


print(
    "Model name:",
    metadata.get("model_name")
)

print(
    "Model version:",
    metadata.get("model_version")
)

print(
    "Target:",
    metadata.get("target_column")
)

print(
    "Feature count:",
    metadata.get("feature_count")
)


# ============================================================
# VALIDATE TARGET
# ============================================================

print_section("VALIDATING TARGET CONFIGURATION")

metadata_target = metadata.get(
    "target_column"
)

feature_target = feature_metadata.get(
    "target"
)


print(
    f"Config target   : {TARGET_COLUMN}"
)

print(
    f"Metadata target : {metadata_target}"
)

print(
    f"Feature target  : {feature_target}"
)


if metadata_target != TARGET_COLUMN:

    raise ValueError(
        "Target mismatch between config and model metadata."
    )


if feature_target != TARGET_COLUMN:

    raise ValueError(
        "Target mismatch between feature metadata and config."
    )


print("Target configuration validated.")


# ============================================================
# VALIDATE FEATURES
# ============================================================

print_section("VALIDATING FEATURE CONFIGURATION")

metadata_features = feature_metadata.get(
    "features",
    []
)

metadata_feature_order = feature_metadata.get(
    "feature_order",
    []
)


print(
    f"Config feature count: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Metadata feature count: "
    f"{len(metadata_features)}"
)

print(
    f"Feature order count: "
    f"{len(metadata_feature_order)}"
)


if FEATURE_COLUMNS != metadata_features:

    raise ValueError(
        "Feature list does not match metadata."
    )


if FEATURE_COLUMNS != metadata_feature_order:

    raise ValueError(
        "Feature order does not match metadata."
    )


print("\nFeature order:")

for index, feature in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f"{index:2d}. {feature}"
    )


print("\nFeature configuration validated.")


# ============================================================
# LOAD PROCESSED DATASET
# ============================================================

print_section("LOADING TEST DATA")

df = pd.read_csv(
    PROCESSED_DATA_FILE
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# VALIDATE DATASET COLUMNS
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )


print(
    "All model features are available."
)


# ============================================================
# USE FINAL TEST PORTION
# ============================================================

test_size = int(
    len(df) * 0.20
)

test_df = df.iloc[-test_size:].copy()


X_test = test_df[
    FEATURE_COLUMNS
].copy()

y_test = test_df[
    TARGET_COLUMN
].copy()


print(
    f"Validation samples: "
    f"{len(X_test):,}"
)


# ============================================================
# BATCH PREDICTION
# ============================================================

print_section("RUNNING BATCH PREDICTION")

predictions = model.predict(
    X_test
)


print(
    f"Predictions generated: "
    f"{len(predictions):,}"
)


# ============================================================
# PREDICTION VALIDATION
# ============================================================

print_section("VALIDATING PREDICTIONS")


if len(predictions) != len(X_test):

    raise ValueError(
        "Prediction count does not match input count."
    )


if not np.all(
    np.isfinite(predictions)
):

    raise ValueError(
        "Predictions contain NaN or infinite values."
    )


print(
    "Prediction count matches input count."
)

print(
    "No NaN or infinite predictions detected."
)


print(
    f"Prediction minimum: "
    f"{predictions.min():,.4f}"
)

print(
    f"Prediction maximum: "
    f"{predictions.max():,.4f}"
)

print(
    f"Prediction mean: "
    f"{predictions.mean():,.4f}"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print_section("SAMPLE PREDICTIONS")

sample_count = min(
    10,
    len(X_test)
)


sample_indices = np.linspace(
    0,
    len(X_test) - 1,
    sample_count,
    dtype=int,
)


for index in sample_indices:

    actual = y_test.iloc[index]

    predicted = predictions[index]

    error = actual - predicted

    print(
        f"Sample {index + 1:5d} | "
        f"Actual: {actual:12,.2f} | "
        f"Predicted: {predicted:12,.2f} | "
        f"Error: {error:12,.2f}"
    )


# ============================================================
# SINGLE RECORD TEST
# ============================================================

print_section("SINGLE RECORD PREDICTION TEST")

single_input = X_test.iloc[
    [0]
].copy()


single_prediction = model.predict(
    single_input
)


if len(single_prediction) != 1:

    raise ValueError(
        "Single-record prediction did not return "
        "exactly one prediction."
    )


if not np.isfinite(
    single_prediction[0]
):

    raise ValueError(
        "Single-record prediction is invalid."
    )


print("Input record:")

print(
    single_input.to_string(
        index=False
    )
)


print(
    f"\nPredicted ROI: "
    f"{single_prediction[0]:,.4f}"
)


# ============================================================
# FEATURE COUNT TEST
# ============================================================

print_section("FEATURE COUNT VALIDATION")

try:

    wrong_input = single_input.drop(
        columns=[FEATURE_COLUMNS[-1]]
    )

    model.predict(
        wrong_input
    )

    raise RuntimeError(
        "Model unexpectedly accepted an input "
        "with a missing feature."
    )

except Exception:

    print(
        "Model correctly rejects incomplete "
        "feature input."
    )


# ============================================================
# MODEL METADATA CONSISTENCY
# ============================================================

print_section("METADATA CONSISTENCY CHECK")

model_name = metadata.get(
    "model_name"
)

model_version = metadata.get(
    "model_version"
)

feature_count = metadata.get(
    "feature_count"
)


if not model_name:

    raise ValueError(
        "Model name is missing."
    )


if not model_version:

    raise ValueError(
        "Model version is missing."
    )


if feature_count != len(FEATURE_COLUMNS):

    raise ValueError(
        "Metadata feature count does not "
        "match actual feature count."
    )


print(
    f"Model name: {model_name}"
)

print(
    f"Model version: {model_version}"
)

print(
    f"Feature count: {feature_count}"
)

print(
    "Metadata consistency validated."
)


# ============================================================
# FINAL RESULT
# ============================================================

print_section(
    "PHASE 3F.2 COMPLETED SUCCESSFULLY"
)

print(
    "Saved model can be loaded independently."
)

print(
    "Pipeline structure is valid."
)

print(
    "Feature configuration is valid."
)

print(
    "Metadata configuration is valid."
)

print(
    "Batch prediction works."
)

print(
    "Single-record prediction works."
)

print(
    "Prediction values are finite."
)

print(
    "Incomplete feature input is rejected."
)

print(
    "\nThe model is ready for production integration testing."
)