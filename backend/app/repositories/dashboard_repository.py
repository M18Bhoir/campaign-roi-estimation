from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.models import Campaign, Prediction


class DashboardRepository:
    """
    Repository responsible for retrieving aggregated
    dashboard analytics from PostgreSQL.
    """

    @staticmethod
    def get_summary(db: Session) -> dict:
        """
        Retrieve high-level dashboard KPI values.
        """

        total_predictions = (
            db.query(func.count(Prediction.id))
            .scalar()
            or 0
        )

        average_predicted_roi = (
            db.query(func.avg(Prediction.predicted_roi))
            .scalar()
        )

        highest_predicted_roi = (
            db.query(func.max(Prediction.predicted_roi))
            .scalar()
        )

        average_budget = (
            db.query(func.avg(Campaign.budget))
            .join(
                Prediction,
                Prediction.campaign_id == Campaign.id,
            )
            .scalar()
        )

        total_campaigns = (
            db.query(func.count(Campaign.id))
            .scalar()
            or 0
        )

        return {
            "total_predictions": int(total_predictions),
            "average_predicted_roi": (
                float(average_predicted_roi)
                if average_predicted_roi is not None
                else 0.0
            ),
            "highest_predicted_roi": (
                float(highest_predicted_roi)
                if highest_predicted_roi is not None
                else 0.0
            ),
            "average_budget": (
                float(average_budget)
                if average_budget is not None
                else 0.0
            ),
            "total_campaigns": int(total_campaigns),
        }

    @staticmethod
    def get_roi_distribution(db: Session) -> list[dict]:
        """
        Retrieve prediction counts grouped by ROI category.

        The current prediction table allows roi_category to be NULL,
        so NULL values are returned as 'Uncategorized'.
        """

        rows = (
            db.query(
                Prediction.roi_category,
                func.count(Prediction.id).label("prediction_count"),
            )
            .group_by(Prediction.roi_category)
            .order_by(func.count(Prediction.id).desc())
            .all()
        )

        return [
            {
                "roi_category": (
                    row.roi_category
                    if row.roi_category is not None
                    else "Uncategorized"
                ),
                "prediction_count": int(row.prediction_count),
            }
            for row in rows
        ]

    @staticmethod
    def get_channel_performance(db: Session) -> list[dict]:
        """
        Retrieve average predicted ROI grouped by marketing channel.
        """

        rows = (
            db.query(
                Campaign.marketing_channel,
                func.count(Prediction.id).label("prediction_count"),
                func.avg(Prediction.predicted_roi).label(
                    "average_predicted_roi"
                ),
            )
            .join(
                Prediction,
                Prediction.campaign_id == Campaign.id,
            )
            .group_by(Campaign.marketing_channel)
            .order_by(
                func.avg(Prediction.predicted_roi).desc()
            )
            .all()
        )

        return [
            {
                "marketing_channel": row.marketing_channel,
                "prediction_count": int(row.prediction_count),
                "average_predicted_roi": (
                    float(row.average_predicted_roi)
                    if row.average_predicted_roi is not None
                    else 0.0
                ),
            }
            for row in rows
        ]

    @staticmethod
    def get_campaign_type_performance(
        db: Session,
    ) -> list[dict]:
        """
        Retrieve average predicted ROI grouped by campaign type.
        """

        rows = (
            db.query(
                Campaign.campaign_type,
                func.count(Prediction.id).label("prediction_count"),
                func.avg(Prediction.predicted_roi).label(
                    "average_predicted_roi"
                ),
            )
            .join(
                Prediction,
                Prediction.campaign_id == Campaign.id,
            )
            .group_by(Campaign.campaign_type)
            .order_by(
                func.avg(Prediction.predicted_roi).desc()
            )
            .all()
        )

        return [
            {
                "campaign_type": row.campaign_type,
                "prediction_count": int(row.prediction_count),
                "average_predicted_roi": (
                    float(row.average_predicted_roi)
                    if row.average_predicted_roi is not None
                    else 0.0
                ),
            }
            for row in rows
        ]

    @staticmethod
    def get_roi_trend(db: Session) -> list[dict]:
        """
        Retrieve average predicted ROI by prediction date.
        """

        rows = (
            db.query(
                func.date(Prediction.created_at).label(
                    "prediction_date"
                ),
                func.count(Prediction.id).label(
                    "prediction_count"
                ),
                func.avg(Prediction.predicted_roi).label(
                    "average_predicted_roi"
                ),
            )
            .group_by(
                func.date(Prediction.created_at)
            )
            .order_by(
                func.date(Prediction.created_at).asc()
            )
            .all()
        )

        return [
            {
                "date": row.prediction_date.isoformat(),
                "prediction_count": int(row.prediction_count),
                "average_predicted_roi": (
                    float(row.average_predicted_roi)
                    if row.average_predicted_roi is not None
                    else 0.0
                ),
            }
            for row in rows
        ]

    @staticmethod
    def get_recent_predictions(
        db: Session,
        limit: int = 10,
    ) -> list[dict]:
        """
        Retrieve the most recent predictions for the dashboard.
        """

        rows = (
            db.query(Prediction)
            .join(
                Campaign,
                Prediction.campaign_id == Campaign.id,
            )
            .order_by(
                Prediction.created_at.desc(),
                Prediction.id.desc(),
            )
            .limit(limit)
            .all()
        )

        return [
            {
                "prediction_id": str(prediction.id),
                "campaign_id": str(prediction.campaign_id),
                "campaign_name": prediction.campaign.campaign_name,
                "campaign_type": prediction.campaign.campaign_type,
                "marketing_channel": (
                    prediction.campaign.marketing_channel
                ),
                "budget": float(prediction.campaign.budget),
                "predicted_roi": float(
                    prediction.predicted_roi
                ),
                "roi_category": prediction.roi_category,
                "model_version": prediction.model_version,
                "created_at": prediction.created_at,
            }
            for prediction in rows
        ]