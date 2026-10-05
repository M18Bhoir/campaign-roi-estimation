from uuid import UUID

from sqlalchemy.orm import Session, joinedload

from backend.app.models import Campaign, Prediction


class PredictionRepository:

    @staticmethod
    def create_prediction(
        db: Session,
        campaign_id: UUID,
        predicted_roi: float,
        predicted_revenue: float | None,
        predicted_profit: float | None,
        roi_category: str | None,
        model_name: str,
        model_version: str,
        prediction_latency_ms: float,
    ) -> Prediction:

        prediction = Prediction(
            campaign_id=campaign_id,
            predicted_roi=predicted_roi,
            predicted_revenue=predicted_revenue,
            predicted_profit=predicted_profit,
            roi_category=roi_category,
            model_name=model_name,
            model_version=model_version,
            prediction_latency_ms=prediction_latency_ms,
        )

        db.add(prediction)

        # Flush without committing.
        db.flush()

        return prediction

    @staticmethod
    def get_prediction_by_id(
        db: Session,
        prediction_id: UUID,
    ) -> Prediction | None:

        return (
            db.query(Prediction)
            .filter(Prediction.id == prediction_id)
            .first()
        )

    @staticmethod
    def get_predictions_by_campaign_id(
        db: Session,
        campaign_id: UUID,
    ) -> list[Prediction]:

        return (
            db.query(Prediction)
            .filter(Prediction.campaign_id == campaign_id)
            .order_by(Prediction.created_at.desc())
            .all()
        )

    @staticmethod
    def get_all_predictions(
        db: Session,
    ) -> list[Prediction]:

        return (
            db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .all()
        )

    @staticmethod
    def get_prediction_with_campaign(
        db: Session,
        prediction_id: UUID,
    ) -> Prediction | None:

        return (
            db.query(Prediction)
            .options(joinedload(Prediction.campaign))
            .filter(Prediction.id == prediction_id)
            .first()
        )
    # ============================================================
    # Paginated Prediction History with Filters
    # ============================================================

    @staticmethod
    def get_predictions_paginated(
        db: Session,
        limit: int,
        offset: int,
        campaign_type: str | None = None,
        marketing_channel: str | None = None,
        model_version: str | None = None,
    ) -> list[Prediction]:


        query = (
            db.query(Prediction)
            .join(
                Campaign,
                Prediction.campaign_id == Campaign.id,
            )
            .options(
            joinedload(Prediction.campaign)
            )
        )

        # --------------------------------------------------------
        # Campaign Type Filter
        # --------------------------------------------------------

        if campaign_type is not None:
            query = query.filter(
                Campaign.campaign_type == campaign_type
            )

        # --------------------------------------------------------
        # Marketing Channel Filter
        # --------------------------------------------------------

        if marketing_channel is not None:
            query = query.filter(
                Campaign.marketing_channel == marketing_channel
            )

        # --------------------------------------------------------
        # Model Version Filter
        # --------------------------------------------------------

        if model_version is not None:
            query = query.filter(
                Prediction.model_version == model_version
            )

        # --------------------------------------------------------
        # Pagination
        # --------------------------------------------------------

        return (
            query
            .order_by(
                Prediction.created_at.desc(),
                Prediction.id.desc(),
            )
            .offset(offset)
            .limit(limit)
            .all()
        )
    

    # ============================================================
    # Count Predictions with Filters
    # ============================================================

    @staticmethod
    def count_predictions(
        db: Session,
        campaign_type: str | None = None,
        marketing_channel: str | None = None,
        model_version: str | None = None,
    ) -> int:

        query = (
            db.query(Prediction)
            .join(
                Campaign,
                Prediction.campaign_id == Campaign.id,
            )
        )

        # --------------------------------------------------------
        # Campaign Type Filter
        # --------------------------------------------------------

        if campaign_type is not None:
            query = query.filter(
                Campaign.campaign_type == campaign_type
            )

        # --------------------------------------------------------
        # Marketing Channel Filter
        # --------------------------------------------------------

        if marketing_channel is not None:
            query = query.filter(
                Campaign.marketing_channel == marketing_channel
            )

        # --------------------------------------------------------
        # Model Version Filter
        # --------------------------------------------------------

        if model_version is not None:
            query = query.filter(
                Prediction.model_version == model_version
            )

        return query.count()