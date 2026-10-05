from sqlalchemy.orm import Session

from backend.app.repositories.dashboard_repository import (
    DashboardRepository,
)


class DashboardService:
    """
    Service layer for dashboard analytics.
    """

    @staticmethod
    def get_summary(db: Session) -> dict:
        return DashboardRepository.get_summary(db)

    @staticmethod
    def get_roi_distribution(
        db: Session,
    ) -> list[dict]:
        return DashboardRepository.get_roi_distribution(db)

    @staticmethod
    def get_channel_performance(
        db: Session,
    ) -> list[dict]:
        return DashboardRepository.get_channel_performance(db)

    @staticmethod
    def get_campaign_type_performance(
        db: Session,
    ) -> list[dict]:
        return DashboardRepository.get_campaign_type_performance(
            db
        )

    @staticmethod
    def get_roi_trend(
        db: Session,
    ) -> list[dict]:
        return DashboardRepository.get_roi_trend(db)

    @staticmethod
    def get_recent_predictions(
        db: Session,
        limit: int = 10,
    ) -> list[dict]:
        return DashboardRepository.get_recent_predictions(
            db,
            limit,
        )