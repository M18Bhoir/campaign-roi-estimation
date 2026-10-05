import time
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

from backend.exceptions import (
    ModelNotReadyError,
    PredictionError,
)

from backend.app.repositories.campaign_repository import (
    CampaignRepository,
)

from backend.app.repositories.prediction_repository import (
    PredictionRepository,
)

from backend.app.services.roi_category import (
    get_roi_category,
)

from ml.src.predictor import predictor


class PredictionService:
    """
    Business/service layer responsible
    for campaign ROI predictions and persistence.
    """

    def __init__(self):
        self.predictor = predictor

    def is_ready(self) -> bool:
        return self.predictor.is_ready()

    # ============================================================
    # Create Prediction
    # ============================================================

    def predict_and_save(
        self,
        db: Session,
        prediction_input: Dict[str, Any],
        campaign_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not self.is_ready():
            raise ModelNotReadyError(
                "Campaign ROI prediction model is not ready."
            )

        try:
            # --------------------------------------------------
            # 1. Generate prediction
            # --------------------------------------------------

            prediction_start = time.perf_counter()

            prediction_result = self.predictor.predict(
                prediction_input
            )

            prediction_latency_ms = (
                time.perf_counter() - prediction_start
            ) * 1000

            predicted_roi = prediction_result["predicted_roi"]
            model_name = prediction_result["model_name"]
            model_version = prediction_result["model_version"]
            roi_category = get_roi_category(predicted_roi)

            # --------------------------------------------------
            # 2. Create campaign
            # --------------------------------------------------

            campaign = CampaignRepository.create_campaign(
                db=db,
                campaign_name=campaign_data["campaign_name"],
                campaign_type=campaign_data["campaign_type"],
                marketing_channel=campaign_data["marketing_channel"],
                target_audience=campaign_data["target_audience"],
                duration_days=campaign_data["duration_days"],
                budget=campaign_data["budget"],
                marketing_spend=None,
                impressions=0,
                clicks=0,
                conversions=0,
                engagement_rate=None,
                ctr=None,
                conversion_rate=None,
                campaign_start_date=campaign_data["campaign_start_date"],
                campaign_end_date=campaign_data["campaign_end_date"],
            )

            # --------------------------------------------------
            # 3. Create prediction
            # --------------------------------------------------

            prediction = PredictionRepository.create_prediction(
                db=db,
                campaign_id=campaign.id,
                predicted_roi=predicted_roi,
                predicted_revenue=None,
                predicted_profit=None,
                roi_category=roi_category,
                model_name=model_name,
                model_version=model_version,
                prediction_latency_ms=prediction_latency_ms,
            )

            # --------------------------------------------------
            # 4. Commit both records together
            # --------------------------------------------------

            db.commit()

            # Refresh after commit so database-generated
            # fields are available.
            db.refresh(campaign)
            db.refresh(prediction)

            return {
                "campaign_id": str(campaign.id),
                "prediction_id": str(prediction.id),
                "predicted_roi": predicted_roi,
                "roi_category": roi_category,
                "model_name": model_name,
                "model_version": model_version,
                "prediction_latency_ms": prediction_latency_ms,
            }

        except (ModelNotReadyError, ValueError):
            db.rollback()
            raise

        except Exception as error:
            db.rollback()

            raise PredictionError(
                "Failed to generate and save ROI prediction."
            ) from error

    # ============================================================
    # Model Information
    # ============================================================

    def get_model_info(
        self,
    ) -> Dict[str, Any]:

        if not self.is_ready():
            raise ModelNotReadyError(
                "Campaign ROI prediction model is not ready."
            )

        return self.predictor.get_model_info()

    # ============================================================
    # Prediction History
    # ============================================================

    def get_prediction_history(
        self,
        db: Session,
        limit: int,
        offset: int,
        campaign_type: str | None = None,
        marketing_channel: str | None = None,
        model_version: str | None = None,
    ) -> Dict[str, Any]:

        predictions = (
            PredictionRepository.get_predictions_paginated(
                db=db,
                limit=limit,
                offset=offset,
                campaign_type=campaign_type,
                marketing_channel=marketing_channel,
                model_version=model_version,
            )
        )

        total = (
            PredictionRepository.count_predictions(
                db=db,
                campaign_type=campaign_type,
                marketing_channel=marketing_channel,
                model_version=model_version,
            )
        )

        history = []

        for prediction in predictions:
            campaign = prediction.campaign

            history.append(
                {
                    "prediction_id": str(prediction.id),
                    "campaign_id": str(prediction.campaign_id),
                    "campaign_name": campaign.campaign_name,
                    "campaign_type": campaign.campaign_type,
                    "marketing_channel": campaign.marketing_channel,
                    "target_audience": campaign.target_audience,
                    "budget": campaign.budget,
                    "duration_days": campaign.duration_days,
                    "predicted_roi": prediction.predicted_roi,
                    "predicted_revenue": prediction.predicted_revenue,
                    "predicted_profit": prediction.predicted_profit,
                    "roi_category": prediction.roi_category,
                    "model_name": prediction.model_name,
                    "model_version": prediction.model_version,
                    "prediction_latency_ms": prediction.prediction_latency_ms,
                    "created_at": prediction.created_at,
                }
            )

        return {
            "data": history,
            "count": len(history),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ============================================================
    # Prediction Detail
    # ============================================================

    def get_prediction_detail(
        self,
        db: Session,
        prediction_id: UUID,
    ) -> Dict[str, Any]:

        prediction = (
            PredictionRepository
            .get_prediction_with_campaign(
                db=db,
                prediction_id=prediction_id,
            )
        )

        if prediction is None:
            raise ValueError(
                f"Prediction with ID '{prediction_id}' "
                "was not found."
            )

        campaign = prediction.campaign

        return {
            "prediction_id": str(
                prediction.id
            ),
            "campaign_id": str(
                prediction.campaign_id
            ),
            "campaign_name": (
                campaign.campaign_name
            ),
            "campaign_type": (
                campaign.campaign_type
            ),
            "marketing_channel": (
                campaign.marketing_channel
            ),
            "target_audience": (
                campaign.target_audience
            ),
            "budget": campaign.budget,
            "duration_days": (
                campaign.duration_days
            ),
            "predicted_roi": (
                prediction.predicted_roi
            ),
            "predicted_revenue": (
                prediction.predicted_revenue
            ),
            "predicted_profit": (
                prediction.predicted_profit
            ),
            "roi_category": (
                prediction.roi_category
            ),
            "model_name": (
                prediction.model_name
            ),
            "model_version": (
                prediction.model_version
            ),
            "prediction_latency_ms": (
                prediction.prediction_latency_ms
            ),
            "created_at": (
                prediction.created_at
            ),
        }


prediction_service = PredictionService()