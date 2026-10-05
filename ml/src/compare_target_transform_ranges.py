import json
import sys
from pathlib import Path

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

from config import PROCESSED_DATA_FILE


RESULTS_DIR = PROJECT_ROOT / "ml" / "results" / "phase_3e"

BASELINE_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "test_predictions.csv"
)

TRANSFORMED_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "transformed_target_predictions.csv"
)

COMPARISON_CSV = (
    RESULTS_DIR
    / "target_transform_range_comparison.csv"
)

COMPARISON_JSON = (
    RESULTS_DIR
    / "target_transform_range_comparison.json"
)


# ============================================================
# DISPLAY
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# ROI RANGE FUNCTION
# ============================================================

def assign_roi_range(value):

    if value < 0:
        return "<0"

    elif value < 100:
        return "0-100"

    elif value < 500:
        return "100-500"

    elif value < 1000:
        return "500-1000"

    elif value < 2000:
        return "1000-2000"

    elif value < 5000:
        return "2000-5000"

    else:
        return ">5000"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print_section("PHASE 3E.3 - TARGET TRANSFORMATION RANGE COMPARISON")

print_section("LOADING PREDICTIONS")

if not BASELINE_PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Baseline predictions not found:\n{BASELINE_PREDICTIONS_FILE}"
    )

if not TRANSFORMED_PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Transformed predictions not found:\n{TRANSFORMED_PREDICTIONS_FILE}"
    )


baseline = pd.read_csv(BASELINE_PREDICTIONS_FILE)
transformed = pd.read_csv(TRANSFORMED_PREDICTIONS_FILE)

print("Baseline predictions shape:", baseline.shape)
print("Transformed predictions shape:", transformed.shape)


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

print_section("PREPARING DATA")


# Baseline file is expected to contain:
# Actual_ROI
# Predicted_ROI

if "Actual_ROI" not in baseline.columns:
    raise ValueError(
        "Baseline predictions must contain 'Actual_ROI'."
    )

if "Predicted_ROI" not in baseline.columns:
    raise ValueError(
        "Baseline predictions must contain 'Predicted_ROI'."
    )


# Transformed file is expected to contain:
# Actual_ROI
# Predicted_ROI

if "Actual_ROI" not in transformed.columns:
    raise ValueError(
        "Transformed predictions must contain 'Actual_ROI'."
    )

if "Predicted_ROI" not in transformed.columns:
    raise ValueError(
        "Transformed predictions must contain 'Predicted_ROI'."
    )


baseline = baseline[
    ["Actual_ROI", "Predicted_ROI"]
].copy()

transformed = transformed[
    ["Actual_ROI", "Predicted_ROI"]
].copy()


baseline = baseline.rename(
    columns={
        "Predicted_ROI": "Baseline_Predicted_ROI"
    }
)

transformed = transformed.rename(
    columns={
        "Predicted_ROI": "Transformed_Predicted_ROI"
    }
)


# ============================================================
# CHECK ACTUAL ROI ALIGNMENT
# ============================================================

print_section("CHECKING ACTUAL ROI ALIGNMENT")

if len(baseline) != len(transformed):
    raise ValueError(
        "Baseline and transformed prediction files have different "
        "numbers of rows."
    )


actual_difference = (
    baseline["Actual_ROI"].values
    - transformed["Actual_ROI"].values
)

max_difference = np.max(np.abs(actual_difference))

print(
    f"Maximum difference between Actual_ROI values: "
    f"{max_difference:.10f}"
)

if max_difference > 1e-8:
    raise ValueError(
        "Actual ROI values are not aligned between the two prediction files."
    )

print("Actual ROI alignment verified.")


# ============================================================
# CREATE COMPARISON DATASET
# ============================================================

comparison = pd.DataFrame({
    "Actual_ROI": baseline["Actual_ROI"],
    "Baseline_Predicted_ROI": baseline["Baseline_Predicted_ROI"],
    "Transformed_Predicted_ROI": transformed[
        "Transformed_Predicted_ROI"
    ],
})


# ============================================================
# CALCULATE ERRORS
# ============================================================

comparison["Baseline_Absolute_Error"] = np.abs(
    comparison["Actual_ROI"]
    - comparison["Baseline_Predicted_ROI"]
)

comparison["Transformed_Absolute_Error"] = np.abs(
    comparison["Actual_ROI"]
    - comparison["Transformed_Predicted_ROI"]
)


comparison["Baseline_Signed_Error"] = (
    comparison["Actual_ROI"]
    - comparison["Baseline_Predicted_ROI"]
)

comparison["Transformed_Signed_Error"] = (
    comparison["Actual_ROI"]
    - comparison["Transformed_Predicted_ROI"]
)


# ============================================================
# ROI RANGES
# ============================================================

comparison["ROI_Range"] = comparison["Actual_ROI"].apply(
    assign_roi_range
)


ROI_RANGE_ORDER = [
    "<0",
    "0-100",
    "100-500",
    "500-1000",
    "1000-2000",
    "2000-5000",
    ">5000",
]


# ============================================================
# RANGE ANALYSIS
# ============================================================

print_section("ROI RANGE PERFORMANCE COMPARISON")


range_results = []


for roi_range in ROI_RANGE_ORDER:

    subset = comparison[
        comparison["ROI_Range"] == roi_range
    ]

    if len(subset) == 0:
        continue

    baseline_mae = subset[
        "Baseline_Absolute_Error"
    ].mean()

    transformed_mae = subset[
        "Transformed_Absolute_Error"
    ].mean()

    baseline_rmse = np.sqrt(
        np.mean(
            subset["Baseline_Signed_Error"] ** 2
        )
    )

    transformed_rmse = np.sqrt(
        np.mean(
            subset["Transformed_Signed_Error"] ** 2
        )
    )

    baseline_bias = subset[
        "Baseline_Signed_Error"
    ].mean()

    transformed_bias = subset[
        "Transformed_Signed_Error"
    ].mean()

    baseline_mean_prediction = subset[
        "Baseline_Predicted_ROI"
    ].mean()

    transformed_mean_prediction = subset[
        "Transformed_Predicted_ROI"
    ].mean()

    actual_mean = subset[
        "Actual_ROI"
    ].mean()

    result = {
        "ROI_Range": roi_range,
        "Samples": int(len(subset)),
        "Actual_Mean_ROI": float(actual_mean),

        "Baseline_Mean_Prediction": float(
            baseline_mean_prediction
        ),

        "Transformed_Mean_Prediction": float(
            transformed_mean_prediction
        ),

        "Baseline_MAE": float(baseline_mae),
        "Transformed_MAE": float(transformed_mae),

        "Baseline_RMSE": float(baseline_rmse),
        "Transformed_RMSE": float(transformed_rmse),

        "Baseline_Bias": float(baseline_bias),
        "Transformed_Bias": float(transformed_bias),
    }

    range_results.append(result)

    print(f"\nROI Range: {roi_range}")
    print(f"Samples: {len(subset):,}")
    print(f"Actual Mean ROI: {actual_mean:,.2f}")

    print(
        f"Baseline Mean Prediction: "
        f"{baseline_mean_prediction:,.2f}"
    )

    print(
        f"Transformed Mean Prediction: "
        f"{transformed_mean_prediction:,.2f}"
    )

    print(
        f"Baseline MAE: "
        f"{baseline_mae:,.2f}"
    )

    print(
        f"Transformed MAE: "
        f"{transformed_mae:,.2f}"
    )

    print(
        f"Baseline RMSE: "
        f"{baseline_rmse:,.2f}"
    )

    print(
        f"Transformed RMSE: "
        f"{transformed_rmse:,.2f}"
    )

    print(
        f"Baseline Bias: "
        f"{baseline_bias:,.2f}"
    )

    print(
        f"Transformed Bias: "
        f"{transformed_bias:,.2f}"
    )


# ============================================================
# EXTREME ROI ANALYSIS
# ============================================================

print_section("EXTREME ROI ANALYSIS")


p99 = comparison["Actual_ROI"].quantile(0.99)
p995 = comparison["Actual_ROI"].quantile(0.995)
p999 = comparison["Actual_ROI"].quantile(0.999)


print(f"99th percentile : {p99:,.2f}")
print(f"99.5th percentile: {p995:,.2f}")
print(f"99.9th percentile: {p999:,.2f}")


extreme = comparison[
    comparison["Actual_ROI"] >= p99
].copy()


print(f"\nExtreme records >= P99: {len(extreme):,}")


if len(extreme) > 0:

    baseline_extreme_mae = (
        extreme["Baseline_Absolute_Error"].mean()
    )

    transformed_extreme_mae = (
        extreme["Transformed_Absolute_Error"].mean()
    )

    baseline_extreme_rmse = np.sqrt(
        np.mean(
            extreme["Baseline_Signed_Error"] ** 2
        )
    )

    transformed_extreme_rmse = np.sqrt(
        np.mean(
            extreme["Transformed_Signed_Error"] ** 2
        )
    )

    print(
        f"Baseline extreme MAE: "
        f"{baseline_extreme_mae:,.2f}"
    )

    print(
        f"Transformed extreme MAE: "
        f"{transformed_extreme_mae:,.2f}"
    )

    print(
        f"Baseline extreme RMSE: "
        f"{baseline_extreme_rmse:,.2f}"
    )

    print(
        f"Transformed extreme RMSE: "
        f"{transformed_extreme_rmse:,.2f}"
    )


# ============================================================
# OVERALL COMPARISON
# ============================================================

print_section("OVERALL RANGE COMPARISON")


range_df = pd.DataFrame(range_results)


if len(range_df) > 0:

    range_df["MAE_Improvement"] = (
        range_df["Baseline_MAE"]
        - range_df["Transformed_MAE"]
    )

    range_df["RMSE_Improvement"] = (
        range_df["Baseline_RMSE"]
        - range_df["Transformed_RMSE"]
    )

    range_df["MAE_Improvement_Percent"] = np.where(
        range_df["Baseline_MAE"] != 0,
        (
            range_df["MAE_Improvement"]
            / range_df["Baseline_MAE"]
        ) * 100,
        0,
    )

    range_df["RMSE_Improvement_Percent"] = np.where(
        range_df["Baseline_RMSE"] != 0,
        (
            range_df["RMSE_Improvement"]
            / range_df["Baseline_RMSE"]
        ) * 100,
        0,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

range_df.to_csv(
    COMPARISON_CSV,
    index=False
)


summary = {
    "baseline_model": {
        "test_r2": 0.3027,
        "test_mae": 2313.3110,
        "test_rmse": 6240.2735,
        "cv_r2": 0.2872,
    },

    "transformed_model": {
        "test_r2": -0.0263,
        "test_mae": 2161.0711,
        "test_rmse": 7570.6198,
        "cv_r2": 0.1245,
    },

    "roi_range_results": range_results,

    "extreme_roi": {
        "p99": float(p99),
        "p995": float(p995),
        "p999": float(p999),
        "sample_count": int(len(extreme)),

        "baseline_mae": float(
            baseline_extreme_mae
        ) if len(extreme) > 0 else None,

        "transformed_mae": float(
            transformed_extreme_mae
        ) if len(extreme) > 0 else None,

        "baseline_rmse": float(
            baseline_extreme_rmse
        ) if len(extreme) > 0 else None,

        "transformed_rmse": float(
            transformed_extreme_rmse
        ) if len(extreme) > 0 else None,
    },
}


with open(
    COMPARISON_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print_section("PHASE 3E.3 COMPLETED")

print(
    f"Range comparison saved:\n{COMPARISON_CSV}"
)

print(
    f"Summary saved:\n{COMPARISON_JSON}"
)