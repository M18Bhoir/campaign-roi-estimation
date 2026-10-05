from datetime import date, datetime, timezone

from backend.app.models import (
    Campaign,
    Prediction,
    ModelMetadata,
)


def test_campaign_model():
    campaign = Campaign(
        campaign_name="Test Campaign",
        campaign_type="Social Media",
        marketing_channel="Instagram",
        target_audience="Young Adults",
        duration_days=30,
        marketing_spend=50000.0,
        impressions=100000,
        clicks=5000,
        conversions=500,
        engagement_rate=5.0,
        ctr=5.0,
        conversion_rate=10.0,
        campaign_start_date=date(2026, 1, 1),
        campaign_end_date=date(2026, 1, 30),
    )

    assert campaign.campaign_name == "Test Campaign"
    assert campaign.marketing_spend == 50000.0
    assert campaign.duration_days == 30


def test_prediction_model():
    prediction = Prediction(
        predicted_roi=2500.50,
        predicted_revenue=None,
        predicted_profit=None,
        roi_category="High",
        model_name="Campaign ROI Random Forest",
        model_version="1.0.0",
        prediction_latency_ms=45.5,
    )

    assert prediction.predicted_roi == 2500.50
    assert prediction.model_version == "1.0.0"


def test_model_metadata():
    metadata = ModelMetadata(
        model_name="Campaign ROI Random Forest",
        version="1.0.0",
        algorithm="RandomForestRegressor",
        test_r2=0.3027,
        cv_r2=0.2872,
        mae=2313.31,
        rmse=6240.27,
        training_samples=120000,
        feature_count=16,
        trained_at=datetime.now(timezone.utc),
        is_active=True,
    )

    assert metadata.model_name == "Campaign ROI Random Forest"
    assert metadata.feature_count == 16
    assert metadata.is_active is True