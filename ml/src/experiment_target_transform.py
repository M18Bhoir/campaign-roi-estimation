import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

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


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "phase_3e"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# OUTPUT FILES
# ================================================================

RESULTS_FILE = (
    RESULTS_DIR
    / "target_transform_comparison.csv"
)

DETAILS_FILE = (
    RESULTS_DIR
    / "target_transform_results.json"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "transformed_target_predictions.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "campaign_roi_model_transformed.joblib"
)


# ================================================================
# UTILITY
# ================================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# SIGNED LOG TRANSFORMATION
# ================================================================

def signed_log_transform(y):

    y = np.asarray(y, dtype=float)

    return (
        np.sign(y)
        * np.log1p(np.abs(y))
    )


# ================================================================
# INVERSE SIGNED LOG TRANSFORMATION
# ================================================================

def inverse_signed_log_transform(y):

    y = np.asarray(y, dtype=float)

    return (
        np.sign(y)
        * np.expm1(np.abs(y))
    )


# ================================================================
# LOAD DATA
# ================================================================

def load_data():

    print_section("LOADING PROCESSED DATASET")

    if not PROCESSED_DATA_FILE.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n"
            f"{PROCESSED_DATA_FILE}"
        )

    df = pd.read_csv(
        PROCESSED_DATA_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[
        required_columns
    ].copy()

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    print(
        f"Final dataset shape: "
        f"{df.shape}"
    )

    return df


# ================================================================
# CREATE MODEL
# ================================================================

def create_model():

    preprocessor = (
        create_preprocessor()
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    return pipeline


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    actual,
    predicted
):

    r2 = r2_score(
        actual,
        predicted
    )

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


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def main():

    print()
    print("=" * 70)
    print(
        "PHASE 3E.2 - SIGNED LOG TARGET "
        "TRANSFORMATION EXPERIMENT"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    df = load_data()

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y_original = df[
        TARGET_COLUMN
    ].astype(float)

    # ------------------------------------------------------------
    # TRANSFORM TARGET
    # ------------------------------------------------------------

    y_transformed = (
        signed_log_transform(
            y_original
        )
    )

    print_section(
        "TARGET TRANSFORMATION"
    )

    print(
        "Transformation:"
    )

    print(
        "sign(ROI) * log1p(abs(ROI))"
    )

    print()

    print(
        f"Original target min: "
        f"{y_original.min():,.4f}"
    )

    print(
        f"Original target max: "
        f"{y_original.max():,.4f}"
    )

    print(
        f"Transformed target min: "
        f"{y_transformed.min():,.4f}"
    )

    print(
        f"Transformed target max: "
        f"{y_transformed.max():,.4f}"
    )

    # ------------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # ------------------------------------------------------------

    split_index = int(
        len(df) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train_original = (
        y_original.iloc[
            :split_index
        ].copy()
    )

    y_test_original = (
        y_original.iloc[
            split_index:
        ].copy()
    )

    y_train_transformed = (
        y_transformed[
            :split_index
        ]
    )

    y_test_transformed = (
        y_transformed[
            split_index:
        ]
    )

    print_section(
        "DATA SPLIT"
    )

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test):,}"
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = create_model()

    # ------------------------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------------------------

    print_section(
        "TIME SERIES CROSS VALIDATION"
    )

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train_transformed,
        cv=tscv,
        scoring="r2",
        n_jobs=None,
    )

    print(
        "CV R² scores:"
    )

    print(
        np.round(
            cv_scores,
            4
        )
    )

    print(
        f"Mean CV R²: "
        f"{cv_scores.mean():.4f}"
    )

    print(
        f"CV R² Std : "
        f"{cv_scores.std():.4f}"
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print_section(
        "TRAINING TRANSFORMED TARGET MODEL"
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train_transformed
    )

    training_time = (
        time.time()
        - start_time
    )

    # ------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------

    transformed_predictions = (
        model.predict(
            X_test
        )
    )

    predictions = (
        inverse_signed_log_transform(
            transformed_predictions
        )
    )

    # ------------------------------------------------------------
    # METRICS - ORIGINAL SCALE
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        y_test_original,
        predictions
    )

    train_transformed_predictions = (
        model.predict(
            X_train
        )
    )

    train_predictions = (
        inverse_signed_log_transform(
            train_transformed_predictions
        )
    )

    train_metrics = calculate_metrics(
        y_train_original,
        train_predictions
    )

    train_test_gap = (
        train_metrics["r2"]
        - metrics["r2"]
    )

    # ------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------

    print_section(
        "TRANSFORMED TARGET RESULTS"
    )

    print("Train Metrics")
    print("-" * 40)

    print(
        f"R²   : "
        f"{train_metrics['r2']:.4f}"
    )

    print(
        f"MAE  : "
        f"{train_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{train_metrics['rmse']:.4f}"
    )

    print()

    print("Test Metrics")
    print("-" * 40)

    print(
        f"R²   : "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"MAE  : "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{metrics['rmse']:.4f}"
    )

    print()

    print(
        f"Train-Test R² Gap: "
        f"{train_test_gap:.4f}"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )

    # ------------------------------------------------------------
    # PREDICTION FILE
    # ------------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "Actual_ROI": (
                y_test_original
                .to_numpy()
            ),
            "Predicted_ROI": (
                predictions
            ),
            "Transformed_Prediction": (
                transformed_predictions
            ),
        }
    )

    prediction_df[
        "Absolute_Error"
    ] = (
        prediction_df["Actual_ROI"]
        - prediction_df["Predicted_ROI"]
    ).abs()

    prediction_df[
        "Signed_Error"
    ] = (
        prediction_df["Actual_ROI"]
        - prediction_df["Predicted_ROI"]
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    # ------------------------------------------------------------
    # RESULTS TABLE
    # ------------------------------------------------------------

    result = {
        "Model": "Random Forest - Signed Log Target",
        "Mean CV R2": float(
            cv_scores.mean()
        ),
        "CV R2 Std": float(
            cv_scores.std()
        ),
        "Train R2": train_metrics["r2"],
        "Test R2": metrics["r2"],
        "Test MAE": metrics["mae"],
        "Test RMSE": metrics["rmse"],
        "Train-Test Gap": train_test_gap,
        "Training Time Seconds": (
            training_time
        ),
    }

    result_df = pd.DataFrame(
        [result]
    )

    result_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------

    details = {
        "experiment": (
            "Signed Log Target Transformation"
        ),
        "transformation": (
            "sign(ROI) * log1p(abs(ROI))"
        ),
        "inverse_transformation": (
            "sign(y) * expm1(abs(y))"
        ),
        "metrics": result,
        "model_file": str(
            MODEL_FILE
        ),
        "prediction_file": str(
            PREDICTIONS_FILE
        ),
    }

    with open(
        DETAILS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            details,
            file,
            indent=4
        )

    # ------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 3E.2 COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Results saved: {RESULTS_FILE}"
    )

    print(
        f"Predictions saved: "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"Experimental model saved: "
        f"{MODEL_FILE}"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()