"""Model evaluation entry points."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -------------------------------------------------------------------
# PROJECT PATH
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# IMPORT PROJECT CONFIGURATION
# -------------------------------------------------------------------

from config import (
    MODEL_FILE,
    METADATA_FILE,
    FEATURE_METADATA_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
)


# -------------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

TEST_PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"

EVALUATION_DIR = RESULTS_DIR / "phase_3d"

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------------------------------------------------
# OUTPUT FILES
# -------------------------------------------------------------------

SUMMARY_FILE = EVALUATION_DIR / "evaluation_summary.json"

ROI_RANGE_FILE = EVALUATION_DIR / "roi_range_performance.csv"

WORST_PREDICTIONS_FILE = EVALUATION_DIR / "worst_predictions.csv"

FEATURE_IMPORTANCE_FILE = (
    EVALUATION_DIR / "feature_importance.csv"
)

AGGREGATED_FEATURE_IMPORTANCE_FILE = (
    EVALUATION_DIR / "aggregated_feature_importance.csv"
)


# -------------------------------------------------------------------
# PLOT FILES
# -------------------------------------------------------------------

ACTUAL_VS_PREDICTED_PLOT = (
    EVALUATION_DIR / "actual_vs_predicted.png"
)

RESIDUAL_PLOT = (
    EVALUATION_DIR / "residual_analysis.png"
)

ERROR_DISTRIBUTION_PLOT = (
    EVALUATION_DIR / "error_distribution.png"
)

ROI_RANGE_PLOT = (
    EVALUATION_DIR / "performance_by_roi_range.png"
)

FEATURE_IMPORTANCE_PLOT = (
    EVALUATION_DIR / "feature_importance.png"
)


# -------------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------------

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------

def load_model():
    print_section("LOADING TRAINED MODEL")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")
    print(f"Model path: {MODEL_FILE}")

    return model


# -------------------------------------------------------------------
# LOAD TEST PREDICTIONS
# -------------------------------------------------------------------

def load_test_predictions():
    print_section("LOADING TEST PREDICTIONS")

    if not TEST_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Test predictions file not found:\n"
            f"{TEST_PREDICTIONS_FILE}"
        )

    df = pd.read_csv(TEST_PREDICTIONS_FILE)

    required_columns = [
        "Actual_ROI",
        "Predicted_ROI",
        "Absolute_Error",
        "Signed_Error",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in test_predictions.csv: "
            f"{missing_columns}"
        )

    print("Test predictions loaded successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    return df


# -------------------------------------------------------------------
# LOAD MODEL METADATA
# -------------------------------------------------------------------

def load_metadata():
    metadata = {}

    if METADATA_FILE.exists():
        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            metadata["model"] = json.load(file)

    if FEATURE_METADATA_FILE.exists():
        with open(
            FEATURE_METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            metadata["features"] = json.load(file)

    return metadata


# -------------------------------------------------------------------
# BASIC MODEL METRICS
# -------------------------------------------------------------------

def calculate_metrics(df):
    actual = df["Actual_ROI"]
    predicted = df["Predicted_ROI"]

    r2 = r2_score(actual, predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }


# -------------------------------------------------------------------
# ACTUAL VS PREDICTED PLOT
# -------------------------------------------------------------------

def create_actual_vs_predicted_plot(df):
    print_section("ACTUAL VS PREDICTED ROI")

    actual = df["Actual_ROI"]
    predicted = df["Predicted_ROI"]

    plt.figure(figsize=(10, 7))

    plt.scatter(
        actual,
        predicted,
        alpha=0.35,
        s=15
    )

    minimum = min(
        actual.min(),
        predicted.min()
    )

    maximum = max(
        actual.max(),
        predicted.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Actual ROI")
    plt.ylabel("Predicted ROI")
    plt.title("Actual vs Predicted ROI")

    plt.tight_layout()

    plt.savefig(
        ACTUAL_VS_PREDICTED_PLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {ACTUAL_VS_PREDICTED_PLOT}"
    )


# -------------------------------------------------------------------
# RESIDUAL ANALYSIS
# -------------------------------------------------------------------

def create_residual_plot(df):
    print_section("RESIDUAL ANALYSIS")

    actual = df["Actual_ROI"]
    predicted = df["Predicted_ROI"]

    residuals = actual - predicted

    plt.figure(figsize=(10, 7))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.35,
        s=15
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Predicted ROI")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Analysis")

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_PLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {RESIDUAL_PLOT}"
    )


# -------------------------------------------------------------------
# ERROR DISTRIBUTION
# -------------------------------------------------------------------

def create_error_distribution_plot(df):
    print_section("ERROR DISTRIBUTION")

    errors = df["Signed_Error"]

    plt.figure(figsize=(10, 7))

    plt.hist(
        errors,
        bins=100,
        alpha=0.75
    )

    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")

    plt.tight_layout()

    plt.savefig(
        ERROR_DISTRIBUTION_PLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {ERROR_DISTRIBUTION_PLOT}"
    )


# -------------------------------------------------------------------
# ROI RANGE ANALYSIS
# -------------------------------------------------------------------

def analyze_roi_ranges(df):
    print_section("ROI RANGE PERFORMANCE ANALYSIS")

    analysis_df = df.copy()

    bins = [
        -np.inf,
        0,
        100,
        500,
        1000,
        2000,
        5000,
        np.inf,
    ]

    labels = [
        "< 0",
        "0 - 100",
        "100 - 500",
        "500 - 1000",
        "1000 - 2000",
        "2000 - 5000",
        "> 5000",
    ]

    analysis_df["ROI_Range"] = pd.cut(
        analysis_df["Actual_ROI"],
        bins=bins,
        labels=labels,
        right=False
    )

    results = []

    for roi_range, group in analysis_df.groupby(
        "ROI_Range",
        observed=False
    ):
        if len(group) == 0:
            continue

        actual = group["Actual_ROI"]
        predicted = group["Predicted_ROI"]

        range_mae = mean_absolute_error(
            actual,
            predicted
        )

        range_rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        range_r2 = r2_score(
            actual,
            predicted
        ) if len(group) > 1 else np.nan

        results.append(
            {
                "ROI_Range": str(roi_range),
                "Samples": int(len(group)),
                "Mean_Actual_ROI": float(actual.mean()),
                "Mean_Predicted_ROI": float(predicted.mean()),
                "MAE": float(range_mae),
                "RMSE": float(range_rmse),
                "R2": (
                    float(range_r2)
                    if not np.isnan(range_r2)
                    else None
                ),
            }
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        ROI_RANGE_FILE,
        index=False
    )

    print()
    print(result_df.to_string(index=False))

    print()
    print(
        f"Saved: {ROI_RANGE_FILE}"
    )

    return result_df


# -------------------------------------------------------------------
# ROI RANGE PLOT
# -------------------------------------------------------------------

def create_roi_range_plot(range_df):
    print_section("ROI RANGE VISUALIZATION")

    plot_df = range_df.copy()

    plt.figure(figsize=(11, 7))

    plt.bar(
        plot_df["ROI_Range"],
        plot_df["MAE"]
    )

    plt.xlabel("Actual ROI Range")
    plt.ylabel("MAE")
    plt.title("Prediction Error by ROI Range")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        ROI_RANGE_PLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {ROI_RANGE_PLOT}"
    )


# -------------------------------------------------------------------
# WORST PREDICTIONS
# -------------------------------------------------------------------

def analyze_worst_predictions(df):
    print_section("WORST PREDICTIONS")

    worst_df = (
        df.sort_values(
            by="Absolute_Error",
            ascending=False
        )
        .head(100)
        .copy()
    )

    worst_df.to_csv(
        WORST_PREDICTIONS_FILE,
        index=False
    )

    print(
        worst_df.head(20).to_string(index=False)
    )

    print()
    print(
        f"Saved top 100 worst predictions to:"
    )
    print(WORST_PREDICTIONS_FILE)

    return worst_df


# -------------------------------------------------------------------
# MODEL PIPELINE COMPONENTS
# -------------------------------------------------------------------

def get_pipeline_components(model):
    """
    Extract the fitted preprocessor and estimator
    from the saved sklearn Pipeline.
    """

    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Saved model is not a sklearn Pipeline."
        )

    named_steps = model.named_steps

    print()
    print("Pipeline steps:")

    for step_name in named_steps:
        print(f"  - {step_name}")

    preprocessor = None
    estimator = None

    for step_name, step in named_steps.items():

        if hasattr(
            step,
            "get_feature_names_out"
        ):
            preprocessor = step

        if hasattr(
            step,
            "feature_importances_"
        ):
            estimator = step

    if preprocessor is None:
        raise ValueError(
            "Could not find fitted preprocessor."
        )

    if estimator is None:
        raise ValueError(
            "Could not find tree-based estimator "
            "with feature_importances_."
        )

    return preprocessor, estimator


# -------------------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------------------

def analyze_feature_importance(model):
    print_section("FEATURE IMPORTANCE")

    preprocessor, estimator = (
        get_pipeline_components(model)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        estimator.feature_importances_
    )

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature name count does not match "
            "feature importance count."
        )

    importance_df = pd.DataFrame(
        {
            "Transformed_Feature": feature_names,
            "Importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df["Importance_Percentage"] = (
        importance_df["Importance"] * 100
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print()
    print(
        "Top 25 transformed features:"
    )

    print(
        importance_df.head(25)
        .to_string(index=False)
    )

    print()
    print(
        f"Saved: {FEATURE_IMPORTANCE_FILE}"
    )

    # ---------------------------------------------------------------
    # AGGREGATE ONE-HOT FEATURES
    # ---------------------------------------------------------------

    aggregated = []

    for _, row in importance_df.iterrows():

        feature_name = row[
            "Transformed_Feature"
        ]

        importance = row[
            "Importance"
        ]

        clean_name = feature_name

        if "__" in clean_name:
            clean_name = (
                clean_name.split(
                    "__",
                    1
                )[1]
            )

        # Handle one-hot encoded categories.
        base_feature = clean_name

        possible_features = [
            "Platform",
            "Region",
            "Device",
            "CustomerSegment",
            "ProductCategory",
            "CampaignType",
            "Season",
            "MarketingObjective",
            "Budget",
            "CompetitorScore",
            "CampaignDurationDays",
            "DiscountPercent",
            "CampaignYear",
            "CampaignMonth",
            "CampaignQuarter",
            "CampaignDayOfWeek",
        ]

        matched = False

        for original_feature in possible_features:

            if (
                clean_name == original_feature
                or clean_name.startswith(
                    original_feature + "_"
                )
            ):
                base_feature = original_feature
                matched = True
                break

        if not matched:
            base_feature = clean_name

        aggregated.append(
            {
                "Original_Feature": base_feature,
                "Importance": importance,
            }
        )

    aggregated_df = (
        pd.DataFrame(aggregated)
        .groupby(
            "Original_Feature",
            as_index=False
        )["Importance"]
        .sum()
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    aggregated_df[
        "Importance_Percentage"
    ] = (
        aggregated_df["Importance"] * 100
    )

    aggregated_df.to_csv(
        AGGREGATED_FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print()
    print(
        "Aggregated original feature importance:"
    )

    print(
        aggregated_df.to_string(index=False)
    )

    print()
    print(
        f"Saved: "
        f"{AGGREGATED_FEATURE_IMPORTANCE_FILE}"
    )

    # ---------------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------------

    top_features = (
        aggregated_df
        .head(15)
        .sort_values(
            "Importance",
            ascending=True
        )
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Original_Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        "Top 15 Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_PLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        f"Saved: {FEATURE_IMPORTANCE_PLOT}"
    )

    return aggregated_df


# -------------------------------------------------------------------
# PREDICTION STATISTICS
# -------------------------------------------------------------------

def calculate_prediction_statistics(df):
    print_section("PREDICTION STATISTICS")

    actual = df["Actual_ROI"]
    predicted = df["Predicted_ROI"]
    error = df["Signed_Error"]

    statistics = {
        "actual": {
            "mean": float(actual.mean()),
            "median": float(actual.median()),
            "std": float(actual.std()),
            "min": float(actual.min()),
            "max": float(actual.max()),
        },
        "predicted": {
            "mean": float(predicted.mean()),
            "median": float(predicted.median()),
            "std": float(predicted.std()),
            "min": float(predicted.min()),
            "max": float(predicted.max()),
        },
        "error": {
            "mean": float(error.mean()),
            "median": float(error.median()),
            "std": float(error.std()),
            "min": float(error.min()),
            "max": float(error.max()),
        },
    }

    print()
    print("Actual ROI")
    print("-" * 40)
    print(f"Mean   : {statistics['actual']['mean']:.4f}")
    print(f"Median : {statistics['actual']['median']:.4f}")
    print(f"Std    : {statistics['actual']['std']:.4f}")
    print(f"Min    : {statistics['actual']['min']:.4f}")
    print(f"Max    : {statistics['actual']['max']:.4f}")

    print()
    print("Predicted ROI")
    print("-" * 40)
    print(
        f"Mean   : "
        f"{statistics['predicted']['mean']:.4f}"
    )
    print(
        f"Median : "
        f"{statistics['predicted']['median']:.4f}"
    )
    print(
        f"Std    : "
        f"{statistics['predicted']['std']:.4f}"
    )
    print(
        f"Min    : "
        f"{statistics['predicted']['min']:.4f}"
    )
    print(
        f"Max    : "
        f"{statistics['predicted']['max']:.4f}"
    )

    print()
    print("Prediction Error")
    print("-" * 40)
    print(
        f"Mean   : "
        f"{statistics['error']['mean']:.4f}"
    )
    print(
        f"Median : "
        f"{statistics['error']['median']:.4f}"
    )
    print(
        f"Std    : "
        f"{statistics['error']['std']:.4f}"
    )

    return statistics


# -------------------------------------------------------------------
# FINAL EVALUATION SUMMARY
# -------------------------------------------------------------------

def create_summary(
    metrics,
    statistics,
    range_df,
    aggregated_importance,
    metadata,
):
    print_section("FINAL EVALUATION SUMMARY")

    summary = {
        "phase": "3D",
        "model": "Random Forest",
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "prediction_statistics": statistics,
        "roi_range_analysis": (
            range_df.to_dict(
                orient="records"
            )
        ),
        "top_features": (
            aggregated_importance
            .head(15)
            .to_dict(
                orient="records"
            )
        ),
        "random_state": RANDOM_STATE,
        "evaluation_status": (
            "Completed"
        ),
    }

    if "model" in metadata:
        summary["training_metadata"] = (
            metadata["model"]
        )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print()
    print(
        f"R²   : {metrics['r2']:.4f}"
    )
    print(
        f"MAE  : {metrics['mae']:.4f}"
    )
    print(
        f"RMSE : {metrics['rmse']:.4f}"
    )

    print()
    print(
        f"Evaluation summary saved:"
    )
    print(SUMMARY_FILE)

    return summary


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("PHASE 3D - MODEL EVALUATION & EXPLAINABILITY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------

    model = load_model()

    predictions_df = load_test_predictions()

    metadata = load_metadata()

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    print_section("MODEL METRICS")

    metrics = calculate_metrics(
        predictions_df
    )

    print(
        f"R²   : {metrics['r2']:.4f}"
    )

    print(
        f"MAE  : {metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {metrics['rmse']:.4f}"
    )

    # ---------------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------------

    create_actual_vs_predicted_plot(
        predictions_df
    )

    create_residual_plot(
        predictions_df
    )

    create_error_distribution_plot(
        predictions_df
    )

    # ---------------------------------------------------------------
    # ROI RANGE
    # ---------------------------------------------------------------

    roi_range_df = analyze_roi_ranges(
        predictions_df
    )

    create_roi_range_plot(
        roi_range_df
    )

    # ---------------------------------------------------------------
    # WORST PREDICTIONS
    # ---------------------------------------------------------------

    analyze_worst_predictions(
        predictions_df
    )

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    aggregated_importance = (
        analyze_feature_importance(
            model
        )
    )

    # ---------------------------------------------------------------
    # PREDICTION STATISTICS
    # ---------------------------------------------------------------

    statistics = (
        calculate_prediction_statistics(
            predictions_df
        )
    )

    # ---------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------

    create_summary(
        metrics=metrics,
        statistics=statistics,
        range_df=roi_range_df,
        aggregated_importance=aggregated_importance,
        metadata=metadata,
    )

    # ---------------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 3D COMPLETED")
    print("=" * 70)

    print()
    print("Evaluation files generated:")
    print(f"  - {ACTUAL_VS_PREDICTED_PLOT}")
    print(f"  - {RESIDUAL_PLOT}")
    print(f"  - {ERROR_DISTRIBUTION_PLOT}")
    print(f"  - {ROI_RANGE_PLOT}")
    print(f"  - {FEATURE_IMPORTANCE_PLOT}")
    print(f"  - {ROI_RANGE_FILE}")
    print(f"  - {WORST_PREDICTIONS_FILE}")
    print(f"  - {FEATURE_IMPORTANCE_FILE}")
    print(
        f"  - {AGGREGATED_FEATURE_IMPORTANCE_FILE}"
    )
    print(f"  - {SUMMARY_FILE}")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()