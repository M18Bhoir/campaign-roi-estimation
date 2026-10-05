from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# Prediction Request
# ============================================================

class CampaignPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Application / Database fields
    campaign_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    marketing_channel: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    target_audience: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    campaign_start_date: date
    campaign_end_date: date

    # ML features
    budget: float = Field(..., ge=0)

    competitor_score: float

    campaign_duration_days: int = Field(
        ...,
        gt=0,
    )

    discount_percent: float = Field(
        ...,
        ge=0,
    )

    platform: str = Field(
        ...,
        min_length=1,
    )

    region: str = Field(
        ...,
        min_length=1,
    )

    device: str = Field(
        ...,
        min_length=1,
    )

    customer_segment: str = Field(
        ...,
        min_length=1,
    )

    product_category: str = Field(
        ...,
        min_length=1,
    )

    campaign_type: str = Field(
        ...,
        min_length=1,
    )

    season: str = Field(
        ...,
        min_length=1,
    )

    marketing_objective: str = Field(
        ...,
        min_length=1,
    )

    campaign_year: int = Field(
        ...,
        ge=2023,
    )

    campaign_month: int = Field(
        ...,
        ge=1,
        le=12,
    )

    campaign_quarter: int = Field(
        ...,
        ge=1,
        le=4,
    )

    campaign_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
    )

    @model_validator(mode="after")
    def validate_campaign_request(self):
    # -------------------------------------------------
    # Date validation
    # -------------------------------------------------
        if self.campaign_end_date < self.campaign_start_date:
             ValueError(
                "campaign_end_date must be greater than or equal to campaign_start_date."
            )

        calculated_duration = (
            self.campaign_end_date - self.campaign_start_date
        ).days + 1

        if self.campaign_duration_days != calculated_duration:
            raise ValueError(
                "campaign_duration_days must match the number of days "
                "between campaign_start_date and campaign_end_date."
            )

    # -------------------------------------------------
    # Budget validation
    # -------------------------------------------------
        if self.budget < 0:
                raise ValueError(
                    "budget must be greater than or equal to 0."
            )   

    # -------------------------------------------------
    # Competitor score validation
    # -------------------------------------------------
        if not 0 <= self.competitor_score <= 100:
            raise ValueError(
                "competitor_score must be between 0 and 100."
            )

    # -------------------------------------------------
    # Discount validation
    # -------------------------------------------------
        if not 0 <= self.discount_percent <= 100:
            raise ValueError(
                "discount_percent must be between 0 and 100."
            )

        return self

# ============================================================
# Prediction Data
# ============================================================

class PredictionData(BaseModel):
    campaign_id: str
    prediction_id: str
    predicted_roi: float
    roi_category: str
    model_name: str
    model_version: str
    prediction_latency_ms: float


# ============================================================
# Standard Success Response
# ============================================================

class PredictionResponse(BaseModel):
    success: bool = True
    data: PredictionData
    request_id: str
    timestamp: datetime


# ============================================================
# Prediction History
# ============================================================

class PredictionHistoryItem(BaseModel):
    prediction_id: str
    campaign_id: str
    campaign_name: str
    campaign_type: str
    marketing_channel: str
    target_audience: str
    budget: float
    duration_days: int
    predicted_roi: float
    predicted_revenue: float | None
    predicted_profit: float | None
    roi_category: str | None
    model_name: str
    model_version: str
    prediction_latency_ms: float
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    success: bool = True
    data: list[PredictionHistoryItem]
    count: int
    total: int
    limit: int
    offset: int
    request_id: str
    timestamp: datetime

class PredictionDetailResponse(BaseModel):
    success: bool = True
    data: PredictionHistoryItem
    request_id: str
    timestamp: datetime


# ============================================================
# Error Details
# ============================================================

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


# ============================================================
# Standard Error Response
# ============================================================

class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
    request_id: str
    timestamp: datetime

# ============================================================
# MODEL METADATA SCHEMAS
# ============================================================

class ModelMetadataData(BaseModel):
    model_id: str
    model_name: str
    version: str
    algorithm: str
    test_r2: float
    cv_r2: float
    mae: float
    rmse: float
    training_samples: int
    feature_count: int
    trained_at: datetime
    is_active: bool


class ModelMetadataResponse(BaseModel):
    success: bool = True
    data: ModelMetadataData
    request_id: str
    timestamp: datetime


# ============================================================
# DASHBOARD SCHEMAS
# ============================================================


class DashboardSummaryData(BaseModel):
    total_predictions: int
    average_predicted_roi: float
    highest_predicted_roi: float
    average_budget: float
    total_campaigns: int


class DashboardSummaryResponse(BaseModel):
    success: bool = True
    data: DashboardSummaryData
    request_id: str
    timestamp: datetime


class ROIDistributionItem(BaseModel):
    roi_category: str
    prediction_count: int


class ROIDistributionResponse(BaseModel):
    success: bool = True
    data: list[ROIDistributionItem]
    request_id: str
    timestamp: datetime


class ChannelPerformanceItem(BaseModel):
    marketing_channel: str
    prediction_count: int
    average_predicted_roi: float


class ChannelPerformanceResponse(BaseModel):
    success: bool = True
    data: list[ChannelPerformanceItem]
    request_id: str
    timestamp: datetime


class CampaignTypePerformanceItem(BaseModel):
    campaign_type: str
    prediction_count: int
    average_predicted_roi: float


class CampaignTypePerformanceResponse(BaseModel):
    success: bool = True
    data: list[CampaignTypePerformanceItem]
    request_id: str
    timestamp: datetime


class ROITrendItem(BaseModel):
    date: str
    prediction_count: int
    average_predicted_roi: float


class ROITrendResponse(BaseModel):
    success: bool = True
    data: list[ROITrendItem]
    request_id: str
    timestamp: datetime


class RecentPredictionItem(BaseModel):
    prediction_id: str
    campaign_id: str
    campaign_name: str
    campaign_type: str
    marketing_channel: str
    budget: float
    predicted_roi: float
    roi_category: str | None
    model_version: str
    created_at: datetime


class RecentPredictionsResponse(BaseModel):
    success: bool = True
    data: list[RecentPredictionItem]
    request_id: str
    timestamp: datetime