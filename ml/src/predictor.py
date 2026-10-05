import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_FILE = MODEL_DIR / "campaign_roi_model.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
FEATURE_METADATA_FILE = MODEL_DIR / "feature_metadata.json"


# ============================================================
# MODEL PREDICTOR
# ============================================================

class CampaignROIPredictor:
    """
    Production prediction service for Campaign ROI.

    Responsibilities:
    - Load the serialized ML pipeline.
    - Load model metadata.
    - Validate incoming features.
    - Convert input into a DataFrame.
    - Generate ROI prediction.
    """

    def __init__(self):

        self.model = None
        self.metadata = None
        self.feature_metadata = None

        self.feature_columns = []
        self.model_version = None
        self.model_name = None

        self._load_artifacts()


    # ========================================================
    # LOAD ARTIFACTS
    # ========================================================

    def _load_artifacts(self) -> None:

        if not MODEL_FILE.exists():

            raise FileNotFoundError(
                f"Model file not found:\n{MODEL_FILE}"
            )

        if not METADATA_FILE.exists():

            raise FileNotFoundError(
                f"Model metadata not found:\n{METADATA_FILE}"
            )

        if not FEATURE_METADATA_FILE.exists():

            raise FileNotFoundError(
                f"Feature metadata not found:\n"
                f"{FEATURE_METADATA_FILE}"
            )


        # Load serialized pipeline

        self.model = joblib.load(
            MODEL_FILE
        )


        # Load model metadata

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)


        # Load feature metadata

        with open(
            FEATURE_METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.feature_metadata = json.load(file)


        # Extract metadata

        self.model_name = self.metadata.get(
            "model_name"
        )

        self.model_version = self.metadata.get(
            "model_version"
        )

        self.feature_columns = self.feature_metadata.get(
            "feature_order",
            []
        )


        # Validate loaded metadata

        if not self.feature_columns:

            raise ValueError(
                "No feature columns found in "
                "feature metadata."
            )


        if self.metadata.get(
            "feature_count"
        ) != len(self.feature_columns):

            raise ValueError(
                "Feature count mismatch between "
                "model metadata and feature metadata."
            )


    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def is_ready(self) -> bool:

        return (
            self.model is not None
            and len(self.feature_columns) > 0
        )


    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    def validate_features(
        self,
        data: Dict[str, Any]
    ) -> None:

        incoming_features = set(
            data.keys()
        )

        required_features = set(
            self.feature_columns
        )


        missing_features = (
            required_features
            - incoming_features
        )


        if missing_features:

            raise ValueError(
                "Missing required features: "
                + ", ".join(
                    sorted(missing_features)
                )
            )


        # We intentionally reject unexpected fields.

        unexpected_features = (
            incoming_features
            - required_features
        )


        if unexpected_features:

            raise ValueError(
                "Unexpected features: "
                + ", ".join(
                    sorted(unexpected_features)
                )
            )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    def _create_dataframe(
        self,
        data: Dict[str, Any]
    ) -> pd.DataFrame:

        self.validate_features(
            data
        )


        dataframe = pd.DataFrame(
            [
                data
            ]
        )


        # Force exact feature order.

        dataframe = dataframe[
            self.feature_columns
        ]


        return dataframe


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:

        dataframe = self._create_dataframe(
            data
        )


        prediction = self.model.predict(
            dataframe
        )


        predicted_roi = float(
            prediction[0]
        )


        return {
            "predicted_roi": predicted_roi,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_model_info(self) -> Dict[str, Any]:

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "algorithm": self.metadata.get(
                "algorithm"
            ),
            "target": self.metadata.get(
                "target_column"
            ),
            "feature_count": self.metadata.get(
                "feature_count"
            ),
            "features": self.feature_columns,
            "test_r2": self.metadata.get(
                "test_r2"
            ),
            "cv_r2": self.metadata.get(
                "cv_r2"
            ),
            "mae": self.metadata.get(
                "mae"
            ),
            "rmse": self.metadata.get(
                "rmse"
            ),
            "production_status": self.metadata.get(
                "production_status"
            ),
        }


# ============================================================
# CREATE SINGLE PREDICTOR INSTANCE
# ============================================================

predictor = CampaignROIPredictor()