"""Data validation entry points."""
from pathlib import Path

import pandas as pd

from config import REQUIRED_COLUMNS, TARGET_COLUMN


def validate_file_exists(file_path: Path) -> None:
    """
    Verify that the dataset exists.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )


def validate_columns(df: pd.DataFrame) -> None:
    """
    Verify that all required columns exist.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_target(df: pd.DataFrame) -> None:
    """
    Validate the ROI target column.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' does not exist."
        )

    if df[TARGET_COLUMN].isna().all():
        raise ValueError(
            "ROI contains no usable values."
        )

    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise TypeError(
            "ROI must be numeric."
        )


def validate_numeric_constraints(df: pd.DataFrame) -> None:
    """
    Validate logical business constraints.
    """

    if (df["Budget"] < 0).any():
        raise ValueError(
            "Budget contains negative values."
        )

    if (df["CampaignDurationDays"] <= 0).any():
        raise ValueError(
            "CampaignDurationDays must be greater than zero."
        )

    if (df["DiscountPercent"] < 0).any():
        raise ValueError(
            "DiscountPercent contains negative values."
        )


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Run all dataset validations.
    """

    validate_columns(df)
    validate_target(df)
    validate_numeric_constraints(df)

    print("Dataset validation completed successfully.")