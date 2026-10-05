import logging
import time
import uuid
import os
from datetime import datetime, timezone
from uuid import UUID

from dotenv import load_dotenv

from fastapi import Depends, FastAPI, HTTPException, Request,Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sqlalchemy.orm import Session

from backend.app.database import get_db

from backend.exceptions import (
    ModelNotReadyError,
    PredictionError,
)

from backend.app.logging_config import (
    configure_logging,
    get_logger,
)

from backend.app.schemas import (
    CampaignPredictionRequest,
    PredictionData,
    PredictionResponse,
    ErrorDetail,
    ErrorResponse,
    PredictionHistoryResponse,
    PredictionDetailResponse,
    ModelMetadataResponse,
    DashboardSummaryResponse,
    ROIDistributionResponse,
    ChannelPerformanceResponse,
    CampaignTypePerformanceResponse,
    ROITrendResponse,
    RecentPredictionsResponse,
)

from backend.app.services.prediction_service import (
    prediction_service,
)

from backend.app.services.model_metadata_service import (
    ModelMetadataService,
)

from backend.app.services.dashboard_service import (
    DashboardService,
)

# ============================================================
# Logging Configuration
# ============================================================

configure_logging()

logger = get_logger(__name__)


# ============================================================
# Environment Configuration
# ============================================================

load_dotenv()

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development",
).strip().lower()

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://127.0.0.1:5173",
).strip()

# ============================================================
# Environment Validation
# ============================================================

VALID_ENVIRONMENTS = {
    "development",
    "production",
}

if ENVIRONMENT not in VALID_ENVIRONMENTS:
    raise RuntimeError(
        "Invalid ENVIRONMENT value. "
        "Expected 'development' or 'production'."
    )

if not FRONTEND_URL:
    raise RuntimeError(
        "FRONTEND_URL environment variable is not set."
    )

# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Campaign ROI Estimation API",
    description=(
        "REST API for predicting campaign ROI "
        "using a machine learning model."
    ),
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if ENVIRONMENT == "development" else None,
)

# ============================================================
# CORS
# ============================================================

if ENVIRONMENT == "development":
    ALLOWED_ORIGINS = [
        FRONTEND_URL,
        "http://localhost:5173",
    ]
else:
    ALLOWED_ORIGINS = [
        FRONTEND_URL,
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# Request ID + Request Logging Middleware
# ============================================================

@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next,
):
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    logger.info(
        "request_started | "
        "request_id=%s | "
        "method=%s | "
        "path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

    except Exception:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "request_failed | "
            "request_id=%s | "
            "method=%s | "
            "path=%s | "
            "latency_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )

        raise

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_completed | "
        "request_id=%s | "
        "method=%s | "
        "path=%s | "
        "status_code=%s | "
        "latency_ms=%.3f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    return response


# ============================================================
# Helper Functions
# ============================================================

def get_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def get_request_id(request: Request) -> str:
    return getattr(
        request.state,
        "request_id",
        str(uuid.uuid4()),
    )


# ============================================================
# Validation Error Handler
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = get_request_id(request)

    logger.warning(
        "validation_error | "
        "request_id=%s | "
        "path=%s | "
        "errors=%s",
        request_id,
        request.url.path,
        exc.errors(),
    )

    errors = []
    
    for error in exc.errors():
            cleaned_error = {
                "type": error.get("type"),
                "loc": error.get("loc"),
                "msg": error.get("msg"),
            }
    
            errors.append(cleaned_error)

    error_response = ErrorResponse(
        success=False,
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=errors,
        ),
        request_id=request_id,
        timestamp=get_timestamp(),
    )

    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(
            mode="json"
        ),
    )

   


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root(request: Request):
    return {
        "success": True,
        "message": "Campaign ROI Estimation API",
        "status": "running",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "request_id": get_request_id(request),
        "timestamp": get_timestamp(),
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/api/health")
def health_check(request: Request):
    return {
        "success": True,
        "status": "healthy",
        "service": "campaign-roi-api",
        "model_ready": prediction_service.is_ready(),
        "request_id": get_request_id(request),
        "timestamp": get_timestamp(),
    }


# ============================================================
# Model Information
# ============================================================

@app.get("/api/model-info")
def model_info(request: Request):

    try:
        model_data = prediction_service.get_model_info()

        return {
            "success": True,
            "data": model_data,
            "request_id": get_request_id(request),
            "timestamp": get_timestamp(),
        }

    except ModelNotReadyError as error:

        request_id = get_request_id(request)

        logger.error(
            "model_not_ready | "
            "request_id=%s | "
            "path=%s | "
            "error=%s",
            request_id,
            request.url.path,
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_NOT_READY",
                message=str(error),
            ),
            request_id=request_id,
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=503,
            content=error_response.model_dump(
                mode="json"
            ),
        )

# ============================================================
# Model Metadata
# ============================================================


@app.get(
    "/api/model/metadata",
    response_model=ModelMetadataResponse,
)
def get_model_metadata(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        model_metadata = (
            ModelMetadataService.get_active_model_metadata(db)
        )

        return ModelMetadataResponse(
            success=True,
            data={
                "model_id": str(model_metadata.id),
                "model_name": model_metadata.model_name,
                "version": model_metadata.version,
                "algorithm": model_metadata.algorithm,
                "test_r2": model_metadata.test_r2,
                "cv_r2": model_metadata.cv_r2,
                "mae": model_metadata.mae,
                "rmse": model_metadata.rmse,
                "training_samples": model_metadata.training_samples,
                "feature_count": model_metadata.feature_count,
                "trained_at": model_metadata.trained_at,
                "is_active": model_metadata.is_active,
            },
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except ModelNotReadyError as error:
        request_id = get_request_id(request)

        logger.error(
            "model_metadata_not_ready | "
            "request_id=%s | "
            "path=%s | "
            "error=%s",
            request_id,
            request.url.path,
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_NOT_READY",
                message=str(error),
            ),
            request_id=request_id,
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=503,
            content=error_response.model_dump(
                mode="json"
            ),
        )

    except Exception as error:
        request_id = get_request_id(request)

        logger.exception(
            "model_metadata_failed | "
            "request_id=%s | "
            "path=%s | "
            "error=%s",
            request_id,
            request.url.path,
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_METADATA_FAILED",
                message="Failed to retrieve model metadata.",
            ),
            request_id=request_id,
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(
                mode="json"
            ),
        )

# ============================================================
# ROI Prediction
# ============================================================

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
)
def predict_roi(
    request: Request,
    prediction_request: CampaignPredictionRequest,
    db: Session = Depends(get_db),
):

    prediction_input = {
        "Budget": prediction_request.budget,

        "CompetitorScore": (
            prediction_request.competitor_score
        ),

        "CampaignDurationDays": (
            prediction_request.campaign_duration_days
        ),

        "DiscountPercent": (
            prediction_request.discount_percent
        ),

        "Platform": prediction_request.platform,

        "Region": prediction_request.region,

        "Device": prediction_request.device,

        "CustomerSegment": (
            prediction_request.customer_segment
        ),

        "ProductCategory": (
            prediction_request.product_category
        ),

        "CampaignType": (
            prediction_request.campaign_type
        ),

        "Season": prediction_request.season,

        "MarketingObjective": (
            prediction_request.marketing_objective
        ),

        "CampaignYear": (
            prediction_request.campaign_year
        ),

        "CampaignMonth": (
            prediction_request.campaign_month
        ),

        "CampaignQuarter": (
            prediction_request.campaign_quarter
        ),

        "CampaignDayOfWeek": (
            prediction_request.campaign_day_of_week
        ),
    }

    campaign_data = {
        "campaign_name": (
            prediction_request.campaign_name
        ),

        "campaign_type": (
            prediction_request.campaign_type
        ),

        "marketing_channel": (
            prediction_request.marketing_channel
        ),

        "target_audience": (
            prediction_request.target_audience
        ),

        "duration_days": (
            prediction_request.campaign_duration_days
        ),

        "budget": (
            prediction_request.budget
        ),

        "campaign_start_date": (
            prediction_request.campaign_start_date
        ),

        "campaign_end_date": (
            prediction_request.campaign_end_date
        ),
    }

    try:

        result = prediction_service.predict_and_save(
            db=db,
            prediction_input=prediction_input,
            campaign_data=campaign_data,
        )

    except ModelNotReadyError as error:

        logger.error(
            "model_not_ready | "
            "request_id=%s | "
            "path=%s | "
            "error=%s",
            get_request_id(request),
            request.url.path,
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_NOT_READY",
                message=str(error),
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=503,
            content=error_response.model_dump(
                mode="json"
            ),
        )

    except ValueError as error:

        logger.warning(
            "invalid_prediction_input | "
            "request_id=%s | "
            "path=%s | "
            "error=%s",
            get_request_id(request),
            request.url.path,
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="INVALID_PREDICTION_INPUT",
                message=str(error),
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=400,
            content=error_response.model_dump(
                mode="json"
            ),
        )

    except PredictionError as error:

        logger.exception(
            "prediction_error | "
            "request_id=%s | "
            "path=%s | "
            "error=%s",
            get_request_id(request),
            request.url.path,
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="PREDICTION_ERROR",
                message=str(error),
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(
                mode="json"
            ),
        )

    logger.info(
        "prediction_completed | "
        "request_id=%s | "
        "model=%s | "
        "model_version=%s | "
        "predicted_roi=%.4f | "
        "prediction_latency_ms=%.3f",
        get_request_id(request),
        result["model_name"],
        result["model_version"],
        result["predicted_roi"],
        result["prediction_latency_ms"],
    )

    return PredictionResponse(
    success=True,
    data=PredictionData(
        campaign_id=result["campaign_id"],
        prediction_id=result["prediction_id"],
        predicted_roi=result["predicted_roi"],
        roi_category=result["roi_category"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        prediction_latency_ms=round(
            result["prediction_latency_ms"],
            3,
        ),
    ),
    request_id=get_request_id(request),
    timestamp=get_timestamp(),
)


# ============================================================
# Prediction History
# ============================================================

@app.get(
    "/api/predictions",
    response_model=PredictionHistoryResponse,
)
def get_prediction_history(
    request: Request,

    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    campaign_type: str | None = Query(
        default=None,
    ),

    marketing_channel: str | None = Query(
        default=None,
    ),

    model_version: str | None = Query(
        default=None,
    ),

    db: Session = Depends(get_db),
):

    try:

        result = prediction_service.get_prediction_history(
            db=db,
            limit=limit,
            offset=offset,
            campaign_type=campaign_type,
            marketing_channel=marketing_channel,
            model_version=model_version,
        )

        return PredictionHistoryResponse(
            success=True,
            data=result["data"],
            count=result["count"],
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except Exception as error:

        logger.exception(
            "prediction_history_failed | "
            "request_id=%s | error=%s",
            get_request_id(request),
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="PREDICTION_HISTORY_FAILED",
                message="Failed to retrieve prediction history.",
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(
                mode="json"
            ),
        )

# ============================================================
# Prediction Detail
# ============================================================

@app.get(
    "/api/predictions/{prediction_id}",
    response_model=PredictionDetailResponse,
)
def get_prediction_detail(
    prediction_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
):

    try:

        prediction = prediction_service.get_prediction_detail(
            db=db,
            prediction_id=prediction_id,
        )

        return PredictionDetailResponse(
            success=True,
            data=prediction,
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except ValueError as error:

        logger.warning(
            "prediction_not_found | "
            "request_id=%s | "
            "prediction_id=%s | "
            "error=%s",
            get_request_id(request),
            prediction_id,
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="PREDICTION_NOT_FOUND",
                message=str(error),
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=404,
            content=error_response.model_dump(
                mode="json"
            ),
        )


# ============================================================
# Dashboard Analytics
# ============================================================


@app.get(
    "/api/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
def get_dashboard_summary(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        result = DashboardService.get_summary(db)

        return DashboardSummaryResponse(
            success=True,
            data=result,
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except Exception as error:
        logger.exception(
            "dashboard_summary_failed | "
            "request_id=%s | error=%s",
            get_request_id(request),
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="DASHBOARD_SUMMARY_FAILED",
                message="Failed to retrieve dashboard summary.",
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json"),
        )


@app.get(
    "/api/dashboard/roi-distribution",
    response_model=ROIDistributionResponse,
)
def get_dashboard_roi_distribution(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        result = DashboardService.get_roi_distribution(db)

        return ROIDistributionResponse(
            success=True,
            data=result,
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except Exception as error:
        logger.exception(
            "dashboard_roi_distribution_failed | "
            "request_id=%s | error=%s",
            get_request_id(request),
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="DASHBOARD_ROI_DISTRIBUTION_FAILED",
                message="Failed to retrieve ROI distribution.",
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json"),
        )


@app.get(
    "/api/dashboard/channel-performance",
    response_model=ChannelPerformanceResponse,
)
def get_dashboard_channel_performance(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        result = DashboardService.get_channel_performance(db)

        return ChannelPerformanceResponse(
            success=True,
            data=result,
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except Exception as error:
        logger.exception(
            "dashboard_channel_performance_failed | "
            "request_id=%s | error=%s",
            get_request_id(request),
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="DASHBOARD_CHANNEL_PERFORMANCE_FAILED",
                message="Failed to retrieve channel performance.",
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json"),
        )


@app.get(
    "/api/dashboard/campaign-type-performance",
    response_model=CampaignTypePerformanceResponse,
)
def get_dashboard_campaign_type_performance(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        result = DashboardService.get_campaign_type_performance(db)

        return CampaignTypePerformanceResponse(
            success=True,
            data=result,
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except Exception as error:
        logger.exception(
            "dashboard_campaign_type_failed | "
            "request_id=%s | error=%s",
            get_request_id(request),
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="DASHBOARD_CAMPAIGN_TYPE_FAILED",
                message="Failed to retrieve campaign type performance.",
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json"),
        )


@app.get(
    "/api/dashboard/roi-trend",
    response_model=ROITrendResponse,
)
def get_dashboard_roi_trend(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        result = DashboardService.get_roi_trend(db)

        return ROITrendResponse(
            success=True,
            data=result,
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except Exception as error:
        logger.exception(
            "dashboard_roi_trend_failed | "
            "request_id=%s | error=%s",
            get_request_id(request),
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="DASHBOARD_ROI_TREND_FAILED",
                message="Failed to retrieve ROI trend.",
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json"),
        )


@app.get(
    "/api/dashboard/recent-predictions",
    response_model=RecentPredictionsResponse,
)
def get_dashboard_recent_predictions(
    request: Request,
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    db: Session = Depends(get_db),
):
    try:
        result = DashboardService.get_recent_predictions(
            db,
            limit,
        )

        return RecentPredictionsResponse(
            success=True,
            data=result,
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

    except Exception as error:
        logger.exception(
            "dashboard_recent_predictions_failed | "
            "request_id=%s | error=%s",
            get_request_id(request),
            str(error),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="DASHBOARD_RECENT_PREDICTIONS_FAILED",
                message="Failed to retrieve recent predictions.",
            ),
            request_id=get_request_id(request),
            timestamp=get_timestamp(),
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json"),
        )