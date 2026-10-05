from datetime import datetime, timezone

from backend.app.models import Campaign, Prediction
from backend.app.repositories.dashboard_repository import (
    DashboardRepository,
)


def create_campaign(
    db,
    campaign_name="Dashboard Test Campaign",
    campaign_type="Sales",
    marketing_channel="Digital",
    budget=100000.0,
):
    campaign = Campaign(
        campaign_name=campaign_name,
        campaign_type=campaign_type,
        marketing_channel=marketing_channel,
        target_audience="Young Adults",
        duration_days=15,
        budget=budget,
        marketing_spend=None,
        impressions=100000,
        clicks=5000,
        conversions=500,
        engagement_rate=5.0,
        ctr=5.0,
        conversion_rate=10.0,
        campaign_start_date=datetime(2026, 10, 1).date(),
        campaign_end_date=datetime(2026, 10, 15).date(),
    )

    db.add(campaign)
    db.flush()

    return campaign


def create_prediction(
    db,
    campaign,
    predicted_roi,
    roi_category=None,
):
    prediction = Prediction(
        campaign_id=campaign.id,
        predicted_roi=predicted_roi,
        predicted_revenue=None,
        predicted_profit=None,
        roi_category=roi_category,
        model_name="Campaign ROI Random Forest",
        model_version="1.0.0",
        prediction_latency_ms=100.0,
        created_at=datetime.now(timezone.utc),
    )

    db.add(prediction)
    db.flush()

    return prediction


def test_get_summary(db_session):
    campaign1 = create_campaign(
        db_session,
        campaign_name="Campaign 1",
        budget=100000.0,
    )

    campaign2 = create_campaign(
        db_session,
        campaign_name="Campaign 2",
        budget=200000.0,
    )

    create_prediction(
        db_session,
        campaign1,
        predicted_roi=100.0,
    )

    create_prediction(
        db_session,
        campaign2,
        predicted_roi=300.0,
    )

    db_session.commit()

    result = DashboardRepository.get_summary(db_session)

    assert result["total_predictions"] == 2
    assert result["total_campaigns"] == 2
    assert result["average_predicted_roi"] == 200.0
    assert result["highest_predicted_roi"] == 300.0
    assert result["average_budget"] == 150000.0


def test_get_roi_distribution(db_session):
    campaign = create_campaign(db_session)

    create_prediction(
        db_session,
        campaign,
        predicted_roi=500.0,
        roi_category="Very High",
    )

    create_prediction(
        db_session,
        campaign,
        predicted_roi=200.0,
        roi_category="Moderate",
    )

    create_prediction(
        db_session,
        campaign,
        predicted_roi=150.0,
        roi_category="Moderate",
    )

    create_prediction(
        db_session,
        campaign,
        predicted_roi=50.0,
        roi_category=None,
    )

    db_session.commit()

    result = DashboardRepository.get_roi_distribution(
        db_session
    )

    distribution = {
        item["roi_category"]: item["prediction_count"]
        for item in result
    }

    assert distribution["Moderate"] == 2
    assert distribution["Very High"] == 1
    assert distribution["Uncategorized"] == 1


def test_get_channel_performance(db_session):
    digital_campaign = create_campaign(
        db_session,
        campaign_name="Digital Campaign",
        marketing_channel="Digital",
    )

    social_campaign = create_campaign(
        db_session,
        campaign_name="Social Campaign",
        marketing_channel="Social Media",
    )

    create_prediction(
        db_session,
        digital_campaign,
        predicted_roi=400.0,
    )

    create_prediction(
        db_session,
        digital_campaign,
        predicted_roi=200.0,
    )

    create_prediction(
        db_session,
        social_campaign,
        predicted_roi=100.0,
    )

    db_session.commit()

    result = DashboardRepository.get_channel_performance(
        db_session
    )

    assert len(result) == 2

    assert result[0]["marketing_channel"] == "Digital"
    assert result[0]["prediction_count"] == 2
    assert result[0]["average_predicted_roi"] == 300.0

    assert result[1]["marketing_channel"] == "Social Media"
    assert result[1]["prediction_count"] == 1
    assert result[1]["average_predicted_roi"] == 100.0


def test_get_campaign_type_performance(db_session):
    sales_campaign = create_campaign(
        db_session,
        campaign_name="Sales Campaign",
        campaign_type="Sales",
    )

    awareness_campaign = create_campaign(
        db_session,
        campaign_name="Awareness Campaign",
        campaign_type="Brand Awareness",
    )

    create_prediction(
        db_session,
        sales_campaign,
        predicted_roi=500.0,
    )

    create_prediction(
        db_session,
        sales_campaign,
        predicted_roi=300.0,
    )

    create_prediction(
        db_session,
        awareness_campaign,
        predicted_roi=100.0,
    )

    db_session.commit()

    result = DashboardRepository.get_campaign_type_performance(
        db_session
    )

    assert len(result) == 2

    assert result[0]["campaign_type"] == "Sales"
    assert result[0]["prediction_count"] == 2
    assert result[0]["average_predicted_roi"] == 400.0

    assert result[1]["campaign_type"] == "Brand Awareness"
    assert result[1]["prediction_count"] == 1
    assert result[1]["average_predicted_roi"] == 100.0


def test_get_roi_trend(db_session):
    campaign = create_campaign(db_session)

    prediction1 = create_prediction(
        db_session,
        campaign,
        predicted_roi=100.0,
    )

    prediction2 = create_prediction(
        db_session,
        campaign,
        predicted_roi=300.0,
    )

    prediction1.created_at = datetime(
        2026,
        10,
        1,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    prediction2.created_at = datetime(
        2026,
        10,
        2,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    db_session.commit()

    result = DashboardRepository.get_roi_trend(
        db_session
    )

    assert len(result) == 2

    assert result[0]["date"] == "2026-10-01"
    assert result[0]["prediction_count"] == 1
    assert result[0]["average_predicted_roi"] == 100.0

    assert result[1]["date"] == "2026-10-02"
    assert result[1]["prediction_count"] == 1
    assert result[1]["average_predicted_roi"] == 300.0


def test_get_recent_predictions(db_session):
    campaign = create_campaign(
        db_session,
        campaign_name="Recent Campaign",
        campaign_type="Sales",
        marketing_channel="Digital",
        budget=150000.0,
    )

    prediction = create_prediction(
        db_session,
        campaign,
        predicted_roi=750.0,
        roi_category="Very High",
    )

    db_session.commit()

    result = DashboardRepository.get_recent_predictions(
        db_session,
        limit=10,
    )

    assert len(result) >= 1

    latest = result[0]

    assert latest["prediction_id"] == str(prediction.id)
    assert latest["campaign_id"] == str(campaign.id)
    assert latest["campaign_name"] == "Recent Campaign"
    assert latest["campaign_type"] == "Sales"
    assert latest["marketing_channel"] == "Digital"
    assert latest["budget"] == 150000.0
    assert latest["predicted_roi"] == 750.0
    assert latest["roi_category"] == "Very High"
    assert latest["model_version"] == "1.0.0"