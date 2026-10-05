from unittest.mock import Mock

from backend.app.repositories.dashboard_repository import (
    DashboardRepository,
)
from backend.app.services.dashboard_service import (
    DashboardService,
)


def test_get_summary():
    db = Mock()

    expected = {
        "total_predictions": 10,
        "average_predicted_roi": 250.0,
        "highest_predicted_roi": 500.0,
        "average_budget": 100000.0,
        "total_campaigns": 8,
    }

    DashboardRepository.get_summary = Mock(
        return_value=expected
    )

    result = DashboardService.get_summary(db)

    assert result == expected
    DashboardRepository.get_summary.assert_called_once_with(db)


def test_get_roi_distribution():
    db = Mock()

    expected = [
        {
            "roi_category": "Very High",
            "prediction_count": 5,
        },
        {
            "roi_category": "Moderate",
            "prediction_count": 3,
        },
    ]

    DashboardRepository.get_roi_distribution = Mock(
        return_value=expected
    )

    result = DashboardService.get_roi_distribution(db)

    assert result == expected
    DashboardRepository.get_roi_distribution.assert_called_once_with(
        db
    )


def test_get_channel_performance():
    db = Mock()

    expected = [
        {
            "marketing_channel": "Digital",
            "prediction_count": 5,
            "average_predicted_roi": 300.0,
        }
    ]

    DashboardRepository.get_channel_performance = Mock(
        return_value=expected
    )

    result = DashboardService.get_channel_performance(db)

    assert result == expected
    DashboardRepository.get_channel_performance.assert_called_once_with(
        db
    )


def test_get_campaign_type_performance():
    db = Mock()

    expected = [
        {
            "campaign_type": "Sales",
            "prediction_count": 4,
            "average_predicted_roi": 350.0,
        }
    ]

    DashboardRepository.get_campaign_type_performance = Mock(
        return_value=expected
    )

    result = DashboardService.get_campaign_type_performance(db)

    assert result == expected
    DashboardRepository.get_campaign_type_performance.assert_called_once_with(
        db
    )


def test_get_roi_trend():
    db = Mock()

    expected = [
        {
            "date": "2026-10-01",
            "prediction_count": 2,
            "average_predicted_roi": 200.0,
        }
    ]

    DashboardRepository.get_roi_trend = Mock(
        return_value=expected
    )

    result = DashboardService.get_roi_trend(db)

    assert result == expected
    DashboardRepository.get_roi_trend.assert_called_once_with(
        db
    )


def test_get_recent_predictions():
    db = Mock()

    expected = [
        {
            "prediction_id": "prediction-1",
            "campaign_id": "campaign-1",
            "campaign_name": "Test Campaign",
            "predicted_roi": 500.0,
        }
    ]

    DashboardRepository.get_recent_predictions = Mock(
        return_value=expected
    )

    result = DashboardService.get_recent_predictions(
        db,
        limit=10,
    )

    assert result == expected

    DashboardRepository.get_recent_predictions.assert_called_once_with(
        db,
        10,
    )