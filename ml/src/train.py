import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from config import (
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DATE_DERIVED_FEATURES,
    MODEL_FILE,
    METADATA_FILE,
    FEATURE_METADATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS,
)

from preprocessing import create_preprocessor


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = MODEL_FILE.parent

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ADDITIONAL OUTPUT FILES
# ============================================================

MODEL_COMPARISON_FILE = (
    RESULTS_DIR / "model_comparison.json"
)

MODEL_COMPARISON_CSV = (
    RESULTS_DIR / "model_comparison.csv"
)

TEST_PREDICTIONS_FILE = (
    RESULTS_DIR / "test_predictions.csv"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    """

    r2 = r2_score(
        y_true,
        y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the processed ML dataset generated in Phase 3B.
    """

    if not PROCESSED_DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n"
            f"{PROCESSED_DATA_FILE}\n\n"
            f"Please run Phase 3B first."
        )

    df = pd.read_csv(
        PROCESSED_DATA_FILE
    )

    print(
        f"Dataset loaded successfully: "
        f"{df.shape}"
    )

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df):
    """
    Validate required features and target.
    """

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if df[TARGET_COLUMN].isnull().any():

        raise ValueError(
            "Target column contains missing values."
        )

    if not np.isfinite(
        df[TARGET_COLUMN]
    ).all():

        raise ValueError(
            "Target column contains "
            "infinite values."
        )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    print(
        "Data validation passed."
    )

    print(
        f"Number of features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Target column: "
        f"{TARGET_COLUMN}"
    )


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():
    """
    Create the candidate regression models.

    Random Forest and XGBoost are regularized compared with
    the previous configuration to reduce overfitting.
    """

    models = {

        # ----------------------------------------------------
        # Linear Regression
        # ----------------------------------------------------

        "Linear Regression": LinearRegression(),

        # ----------------------------------------------------
        # Regularized Random Forest
        # ----------------------------------------------------

        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # Regularized XGBoost
        # ----------------------------------------------------

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def train_models(df):
    """
    Train all candidate models and evaluate them using:

    - TimeSeriesSplit CV R²
    - Train R²
    - Test R²
    - MAE
    - RMSE
    - Train/Test R² gap

    The best model is selected using Mean CV R².
    """

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_index = int(
        len(df) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test):,}"
    )

    print(
        f"Training percentage: "
        f"{len(X_train) / len(df) * 100:.1f}%"
    )

    print(
        f"Testing percentage : "
        f"{len(X_test) / len(df) * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Time-series cross-validation
    # --------------------------------------------------------

    cv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    models = create_models()

    results = {}

    best_model_name = None
    best_cv_score = -np.inf
    best_pipeline = None

    best_test_predictions = None

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for model_name, model in models.items():

        print()
        print("=" * 70)
        print(model_name)
        print("=" * 70)

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        preprocessor = create_preprocessor()

        # ----------------------------------------------------
        # Complete pipeline
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Cross-validation
        # ----------------------------------------------------

        print(
            "Running TimeSeriesSplit..."
        )

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=1,
        )

        mean_cv_r2 = (
            cv_scores.mean()
        )

        std_cv_r2 = (
            cv_scores.std()
        )

        print(
            "CV R² scores:",
            np.round(
                cv_scores,
                4
            )
        )

        print(
            f"Mean CV R²: "
            f"{mean_cv_r2:.4f}"
        )

        print(
            f"CV R² Std: "
            f"{std_cv_r2:.4f}"
        )

        # ----------------------------------------------------
        # Train final pipeline
        # ----------------------------------------------------

        print(
            "Training final model..."
        )

        pipeline.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        train_predictions = (
            pipeline.predict(
                X_train
            )
        )

        test_predictions = (
            pipeline.predict(
                X_test
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        train_metrics = calculate_metrics(
            y_train,
            train_predictions
        )

        test_metrics = calculate_metrics(
            y_test,
            test_predictions
        )

        # ----------------------------------------------------
        # Overfitting gap
        # ----------------------------------------------------

        overfitting_gap = (
            train_metrics["r2"]
            - test_metrics["r2"]
        )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print()
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
            f"{test_metrics['r2']:.4f}"
        )

        print(
            f"MAE  : "
            f"{test_metrics['mae']:.4f}"
        )

        print(
            f"RMSE : "
            f"{test_metrics['rmse']:.4f}"
        )

        print()
        print(
            f"Train-Test R² Gap: "
            f"{overfitting_gap:.4f}"
        )

        print(
            f"Training Time: "
            f"{elapsed_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results[model_name] = {

            "cv_r2_scores":
                [
                    float(score)
                    for score in cv_scores
                ],

            "mean_cv_r2":
                float(mean_cv_r2),

            "std_cv_r2":
                float(std_cv_r2),

            "train_metrics":
                train_metrics,

            "test_metrics":
                test_metrics,

            "overfitting_gap":
                float(overfitting_gap),

            "training_time_seconds":
                float(elapsed_time),
        }

        # ----------------------------------------------------
        # Model selection
        # ----------------------------------------------------

        if mean_cv_r2 > best_cv_score:

            best_cv_score = (
                mean_cv_r2
            )

            best_model_name = (
                model_name
            )

            best_pipeline = (
                pipeline
            )

            best_test_predictions = (
                test_predictions
            )

    # --------------------------------------------------------
    # Test prediction dataframe
    # --------------------------------------------------------

    prediction_comparison = pd.DataFrame({

        "Actual_ROI":
            y_test.to_numpy(),

        "Predicted_ROI":
            best_test_predictions,

    })

    prediction_comparison[
        "Absolute_Error"
    ] = (
        prediction_comparison[
            "Actual_ROI"
        ]
        - prediction_comparison[
            "Predicted_ROI"
        ]
    ).abs()

    prediction_comparison[
        "Signed_Error"
    ] = (
        prediction_comparison[
            "Actual_ROI"
        ]
        - prediction_comparison[
            "Predicted_ROI"
        ]
    )

    return (
        best_model_name,
        best_pipeline,
        results,
        prediction_comparison,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    best_model_name,
    best_pipeline,
    results,
    training_samples,
    prediction_comparison,
):
    """
    Save:

    1. Complete ML pipeline
    2. Model metadata
    3. Feature metadata
    4. Model comparison
    5. Test predictions
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE COMPLETE PIPELINE
    # ========================================================

    joblib.dump(
        best_pipeline,
        MODEL_FILE
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Selected model: "
        f"{best_model_name}"
    )

    print(
        f"Model path: "
        f"{MODEL_FILE}"
    )

    # ========================================================
    # SELECTED MODEL RESULTS
    # ========================================================

    selected_results = (
        results[
            best_model_name
        ]
    )

    # ========================================================
    # MODEL METADATA
    # ========================================================

    metadata = {

        "project":
            "Campaign ROI Estimation",

        "target":
            TARGET_COLUMN,

        "selected_model":
            best_model_name,

        "selection_metric":
            "Mean TimeSeriesSplit CV R²",

        "mean_cv_r2":
            selected_results[
                "mean_cv_r2"
            ],

        "cv_r2_std":
            selected_results[
                "std_cv_r2"
            ],

        "test_r2":
            selected_results[
                "test_metrics"
            ]["r2"],

        "test_mae":
            selected_results[
                "test_metrics"
            ]["mae"],

        "test_rmse":
            selected_results[
                "test_metrics"
            ]["rmse"],

        "train_r2":
            selected_results[
                "train_metrics"
            ]["r2"],

        "train_mae":
            selected_results[
                "train_metrics"
            ]["mae"],

        "train_rmse":
            selected_results[
                "train_metrics"
            ]["rmse"],

        "overfitting_gap":
            selected_results[
                "overfitting_gap"
            ],

        "training_time_seconds":
            selected_results[
                "training_time_seconds"
            ],

        "training_samples":
            training_samples,

        "feature_count":
            len(FEATURE_COLUMNS),

        "numerical_features":
            NUMERICAL_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "date_features":
            DATE_DERIVED_FEATURES,

        "all_features":
            FEATURE_COLUMNS,

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "cv_folds":
            N_SPLITS,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # ========================================================
    # FEATURE METADATA
    # ========================================================

    feature_metadata = {

        "target":
            TARGET_COLUMN,

        "numerical_features":
            NUMERICAL_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "date_derived_features":
            DATE_DERIVED_FEATURES,

        "feature_count":
            len(FEATURE_COLUMNS),

        "all_features":
            FEATURE_COLUMNS,

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
    }

    with open(
        FEATURE_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_metadata,
            file,
            indent=4
        )

    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    with open(
        MODEL_COMPARISON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Create comparison DataFrame
    # --------------------------------------------------------

    comparison_rows = []

    for model_name, model_results in (
        results.items()
    ):

        comparison_rows.append({

            "Model":
                model_name,

            "Mean_CV_R2":
                model_results[
                    "mean_cv_r2"
                ],

            "CV_R2_Std":
                model_results[
                    "std_cv_r2"
                ],

            "Train_R2":
                model_results[
                    "train_metrics"
                ]["r2"],

            "Test_R2":
                model_results[
                    "test_metrics"
                ]["r2"],

            "Train_MAE":
                model_results[
                    "train_metrics"
                ]["mae"],

            "Test_MAE":
                model_results[
                    "test_metrics"
                ]["mae"],

            "Train_RMSE":
                model_results[
                    "train_metrics"
                ]["rmse"],

            "Test_RMSE":
                model_results[
                    "test_metrics"
                ]["rmse"],

            "Overfitting_Gap":
                model_results[
                    "overfitting_gap"
                ],

            "Training_Time_Seconds":
                model_results[
                    "training_time_seconds"
                ],
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df = (
        comparison_df
        .sort_values(
            "Mean_CV_R2",
            ascending=False
        )
        .reset_index(drop=True)
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_CSV,
        index=False
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    prediction_comparison.to_csv(
        TEST_PREDICTIONS_FILE,
        index=False
    )

    # ========================================================
    # PRINT SAVED FILES
    # ========================================================

    print(
        f"Metadata saved: "
        f"{METADATA_FILE}"
    )

    print(
        f"Feature metadata saved: "
        f"{FEATURE_METADATA_FILE}"
    )

    print(
        f"Model comparison saved: "
        f"{MODEL_COMPARISON_CSV}"
    )

    print(
        f"Test predictions saved: "
        f"{TEST_PREDICTIONS_FILE}"
    )


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

def print_final_comparison(results):
    """
    Print a clean comparison table.
    """

    rows = []

    for model_name, model_results in (
        results.items()
    ):

        rows.append({

            "Model":
                model_name,

            "Mean CV R²":
                round(
                    model_results[
                        "mean_cv_r2"
                    ],
                    4
                ),

            "Test R²":
                round(
                    model_results[
                        "test_metrics"
                    ]["r2"],
                    4
                ),

            "Test MAE":
                round(
                    model_results[
                        "test_metrics"
                    ]["mae"],
                    2
                ),

            "Test RMSE":
                round(
                    model_results[
                        "test_metrics"
                    ]["rmse"],
                    2
                ),

            "Train-Test Gap":
                round(
                    model_results[
                        "overfitting_gap"
                    ],
                    4
                ),
        })

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_df = (
        comparison_df
        .sort_values(
            "Mean CV R²",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CAMPAIGN ROI MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_data(df)

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    (
        best_model_name,
        best_pipeline,
        results,
        prediction_comparison,
    ) = train_models(df)

    # --------------------------------------------------------
    # Training sample count
    # --------------------------------------------------------

    training_samples = int(
        len(df) * (1 - TEST_SIZE)
    )

    # --------------------------------------------------------
    # Save everything
    # --------------------------------------------------------

    save_model(
        best_model_name,
        best_pipeline,
        results,
        training_samples,
        prediction_comparison,
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print_final_comparison(
        results
    )

    # --------------------------------------------------------
    # Sample predictions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE TEST PREDICTIONS")
    print("=" * 70)

    print(
        prediction_comparison.head(
            20
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best model: "
        f"{best_model_name}"
    )

    print(
        f"Best Mean CV R²: "
        f"{results[best_model_name]['mean_cv_r2']:.4f}"
    )

    print(
        f"Test R²: "
        f"{results[best_model_name]['test_metrics']['r2']:.4f}"
    )

    print(
        f"Test MAE: "
        f"{results[best_model_name]['test_metrics']['mae']:.4f}"
    )

    print(
        f"Test RMSE: "
        f"{results[best_model_name]['test_metrics']['rmse']:.4f}"
    )

    print(
        f"Train-Test R² Gap: "
        f"{results[best_model_name]['overfitting_gap']:.4f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

