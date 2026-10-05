import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# CONFIG
# ================================================================

from config import (
    RAW_DATA_FILE,
    SHEET_NAME,
    TARGET_COLUMN,
)


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

STATISTICS_FILE = (
    RESULTS_DIR
    / "roi_statistics.json"
)

PERCENTILES_FILE = (
    RESULTS_DIR
    / "roi_percentiles.csv"
)

EXTREME_RECORDS_FILE = (
    RESULTS_DIR
    / "extreme_roi_records.csv"
)

REPORT_FILE = (
    RESULTS_DIR
    / "target_analysis_report.json"
)

DISTRIBUTION_PLOT = (
    RESULTS_DIR
    / "roi_distribution.png"
)

BOXPLOT = (
    RESULTS_DIR
    / "roi_boxplot.png"
)

LOG_DISTRIBUTION_PLOT = (
    RESULTS_DIR
    / "roi_transformed_distribution.png"
)


# ================================================================
# PRINT SECTION
# ================================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# LOAD DATA
# ================================================================

def load_data():

    print_section("LOADING DATASET")

    if not RAW_DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_DATA_FILE}"
        )

    df = pd.read_excel(
        RAW_DATA_FILE,
        sheet_name=SHEET_NAME
    )

    print(
        f"Dataset loaded successfully: {df.shape}"
    )

    print(
        f"Target column: {TARGET_COLUMN}"
    )

    return df


# ================================================================
# VALIDATE TARGET
# ================================================================

def validate_target(df):

    print_section("TARGET VALIDATION")

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "does not exist."
        )

    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    missing = target.isna().sum()

    infinite = np.isinf(target).sum()

    print(
        f"Total records       : {len(target):,}"
    )

    print(
        f"Missing ROI         : {missing:,}"
    )

    print(
        f"Infinite ROI        : {infinite:,}"
    )

    print(
        f"Valid ROI records   : "
        f"{target.notna().sum():,}"
    )

    if missing > 0:

        print(
            "\nWARNING: Missing ROI values detected."
        )

    if infinite > 0:

        print(
            "\nWARNING: Infinite ROI values detected."
        )

    return target


# ================================================================
# BASIC STATISTICS
# ================================================================

def calculate_statistics(target):

    print_section("ROI STATISTICS")

    clean_target = target.dropna()

    statistics = {
        "count": int(clean_target.count()),
        "mean": float(clean_target.mean()),
        "median": float(clean_target.median()),
        "std": float(clean_target.std()),
        "min": float(clean_target.min()),
        "max": float(clean_target.max()),
        "skewness": float(clean_target.skew()),
        "negative_count": int(
            (clean_target < 0).sum()
        ),
        "zero_count": int(
            (clean_target == 0).sum()
        ),
        "positive_count": int(
            (clean_target > 0).sum()
        ),
    }

    total = len(clean_target)

    statistics[
        "negative_percentage"
    ] = (
        statistics["negative_count"]
        / total
        * 100
    )

    statistics[
        "zero_percentage"
    ] = (
        statistics["zero_count"]
        / total
        * 100
    )

    statistics[
        "positive_percentage"
    ] = (
        statistics["positive_count"]
        / total
        * 100
    )

    for key, value in statistics.items():

        if isinstance(value, float):

            print(
                f"{key:25s}: "
                f"{value:,.4f}"
            )

        else:

            print(
                f"{key:25s}: "
                f"{value:,}"
            )

    with open(
        STATISTICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4
        )

    print()
    print(
        f"Saved: {STATISTICS_FILE}"
    )

    return statistics


# ================================================================
# PERCENTILE ANALYSIS
# ================================================================

def analyze_percentiles(target):

    print_section("ROI PERCENTILE ANALYSIS")

    clean_target = target.dropna()

    percentile_values = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
        100,
    ]

    values = np.percentile(
        clean_target,
        percentile_values
    )

    percentile_df = pd.DataFrame(
        {
            "Percentile": percentile_values,
            "ROI": values,
        }
    )

    print(
        percentile_df.to_string(
            index=False
        )
    )

    percentile_df.to_csv(
        PERCENTILES_FILE,
        index=False
    )

    print()
    print(
        f"Saved: {PERCENTILES_FILE}"
    )

    return percentile_df


# ================================================================
# EXTREME VALUE ANALYSIS
# ================================================================

def analyze_extreme_values(
    df,
    target
):

    print_section("EXTREME ROI ANALYSIS")

    analysis_df = df.copy()

    analysis_df[
        "_ROI_NUMERIC"
    ] = pd.to_numeric(
        target,
        errors="coerce"
    )

    clean_target = (
        analysis_df[
            "_ROI_NUMERIC"
        ]
        .dropna()
    )

    # ------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------

    p99 = clean_target.quantile(0.99)
    p995 = clean_target.quantile(0.995)
    p999 = clean_target.quantile(0.999)

    print(
        f"99th percentile  : {p99:,.2f}"
    )

    print(
        f"99.5th percentile: {p995:,.2f}"
    )

    print(
        f"99.9th percentile: {p999:,.2f}"
    )

    print()

    extreme_mask = (
        analysis_df[
            "_ROI_NUMERIC"
        ] >= p99
    )

    extreme_df = (
        analysis_df[
            extreme_mask
        ]
        .sort_values(
            "_ROI_NUMERIC",
            ascending=False
        )
        .copy()
    )

    print(
        f"Records >= 99th percentile: "
        f"{len(extreme_df):,}"
    )

    print()

    columns_to_show = [
        "CampaignID",
        "CampaignDate",
        "Platform",
        "Region",
        "Device",
        "CampaignType",
        "Budget",
        "Spend",
        "Revenue",
        "Profit",
        "ROI",
        "CampaignDurationDays",
        "MarketingObjective",
    ]

    available_columns = [
        column
        for column in columns_to_show
        if column in extreme_df.columns
    ]

    print(
        extreme_df[
            available_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    extreme_df.drop(
        columns=[
            "_ROI_NUMERIC"
        ],
        errors="ignore"
    ).to_csv(
        EXTREME_RECORDS_FILE,
        index=False
    )

    print()
    print(
        f"Saved extreme records to:"
    )

    print(
        EXTREME_RECORDS_FILE
    )

    return {
        "p99": float(p99),
        "p995": float(p995),
        "p999": float(p999),
        "extreme_count": int(
            len(extreme_df)
        ),
    }


# ================================================================
# DISTRIBUTION PLOT
# ================================================================

def create_distribution_plot(target):

    print_section("ROI DISTRIBUTION")

    clean_target = target.dropna()

    plt.figure(
        figsize=(11, 7)
    )

    plt.hist(
        clean_target,
        bins=150,
        alpha=0.75
    )

    plt.xlabel(
        "ROI"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "ROI Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        DISTRIBUTION_PLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {DISTRIBUTION_PLOT}"
    )


# ================================================================
# BOX PLOT
# ================================================================

def create_boxplot(target):

    print_section("ROI BOXPLOT")

    clean_target = target.dropna()

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        clean_target,
        vert=False
    )

    plt.xlabel(
        "ROI"
    )

    plt.title(
        "ROI Boxplot"
    )

    plt.tight_layout()

    plt.savefig(
        BOXPLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {BOXPLOT}"
    )


# ================================================================
# SIGNED LOG TRANSFORMATION
# ================================================================

def create_transformed_distribution(
    target
):

    print_section(
        "SIGNED LOG ROI TRANSFORMATION"
    )

    clean_target = target.dropna()

    transformed = (
        np.sign(clean_target)
        * np.log1p(
            np.abs(clean_target)
        )
    )

    print(
        "Transformation:"
    )

    print(
        "sign(ROI) * log1p(abs(ROI))"
    )

    print()

    print(
        f"Original mean   : "
        f"{clean_target.mean():,.4f}"
    )

    print(
        f"Original median : "
        f"{clean_target.median():,.4f}"
    )

    print(
        f"Transformed mean: "
        f"{transformed.mean():,.4f}"
    )

    print(
        f"Transformed median: "
        f"{transformed.median():,.4f}"
    )

    print(
        f"Transformed min : "
        f"{transformed.min():,.4f}"
    )

    print(
        f"Transformed max : "
        f"{transformed.max():,.4f}"
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.hist(
        transformed,
        bins=100,
        alpha=0.75
    )

    plt.xlabel(
        "Signed Log Transformed ROI"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Signed Log Transformation of ROI"
    )

    plt.tight_layout()

    plt.savefig(
        LOG_DISTRIBUTION_PLOT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        f"Saved: "
        f"{LOG_DISTRIBUTION_PLOT}"
    )

    return transformed


# ================================================================
# TARGET QUALITY REPORT
# ================================================================

def create_report(
    statistics,
    percentile_df,
    extreme_info
):

    print_section(
        "TARGET ANALYSIS REPORT"
    )

    report = {
        "phase": "3E.1",
        "target_column": TARGET_COLUMN,
        "dataset_file": str(
            RAW_DATA_FILE
        ),
        "sheet_name": SHEET_NAME,
        "statistics": statistics,
        "percentiles": percentile_df.to_dict(
            orient="records"
        ),
        "extreme_value_analysis": (
            extreme_info
        ),
        "recommendation_status": (
            "Target distribution analysis completed"
        ),
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"Saved: {REPORT_FILE}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 70)
    print(
        "PHASE 3E.1 - ROI TARGET "
        "DISTRIBUTION & DATA QUALITY ANALYSIS"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    df = load_data()

    # ------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------

    target = validate_target(df)

    # ------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------

    statistics = calculate_statistics(
        target
    )

    # ------------------------------------------------------------
    # PERCENTILES
    # ------------------------------------------------------------

    percentile_df = analyze_percentiles(
        target
    )

    # ------------------------------------------------------------
    # EXTREME VALUES
    # ------------------------------------------------------------

    extreme_info = analyze_extreme_values(
        df,
        target
    )

    # ------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------

    create_distribution_plot(
        target
    )

    create_boxplot(
        target
    )

    # ------------------------------------------------------------
    # TRANSFORMATION ANALYSIS
    # ------------------------------------------------------------

    create_transformed_distribution(
        target
    )

    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------

    create_report(
        statistics,
        percentile_df,
        extreme_info
    )

    # ------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 3E.1 COMPLETED")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        f"  - {STATISTICS_FILE}"
    )

    print(
        f"  - {PERCENTILES_FILE}"
    )

    print(
        f"  - {EXTREME_RECORDS_FILE}"
    )

    print(
        f"  - {DISTRIBUTION_PLOT}"
    )

    print(
        f"  - {BOXPLOT}"
    )

    print(
        f"  - {LOG_DISTRIBUTION_PLOT}"
    )

    print(
        f"  - {REPORT_FILE}"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()