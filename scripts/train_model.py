"""
Full model training pipeline.
Trains XGBoost and Random Forest, compares performance, saves best model.

Usage:
    python scripts/train_model.py --data data/raw/campaign_data.csv
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from backend.app.ml.models.xgboost_model import XGBoostROIModel
from backend.app.ml.models.random_forest_model import RandomForestROIModel
from backend.app.ml.utils.feature_engineering import engineer_features
from backend.app.ml.utils.preprocessor import FeaturePreprocessor
from backend.app.ml.utils.metrics import model_evaluation_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Apply feature engineering to the full dataset."""
    feature_rows = []
    for _, row in df.iterrows():
        feat_df = engineer_features(row.to_dict())
        feature_rows.append(feat_df.iloc[0])
    X = pd.DataFrame(feature_rows)
    y = df["roi_percent"]
    return X, y


def train_and_evaluate(data_path: str):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset: {len(df):,} rows")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Preprocess
    preprocessor = FeaturePreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    results = {}

    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb = XGBoostROIModel({"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05})
    xgb.model_version = "v1.0-xgb"
    xgb.train(X_train_scaled, y_train)
    xgb_preds = xgb.predict(X_test_scaled)
    metrics = model_evaluation_report(y_test.values, xgb_preds)
    results["xgboost"] = metrics
    xgb.r2 = metrics["r2"]
    xgb.mae = metrics["mae"]
    xgb.rmse = metrics["rmse"]
    xgb.mape = metrics["mape"]
    logger.info(f"XGBoost Test Metrics: {results['xgboost']}")

    # Save best model
    Path("data/models").mkdir(parents=True, exist_ok=True)
    xgb.save("data/models/roi_model.pkl")
    preprocessor.save("data/models/preprocessor.pkl")

    # Save evaluation report
    report_df = pd.DataFrame(results).T
    report_df.to_csv("data/models/evaluation_report.csv")
    logger.info("Training complete. Model saved to data/models/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/campaign_data.csv")
    args = parser.parse_args()
    train_and_evaluate(args.data)
