from backend.app.database import SessionLocal
from backend.app.models import Campaign
from backend.app.repositories.prediction_repository import (
    PredictionRepository,
)


def create_test_campaign(db):
    campaign = Campaign(
    campaign_name="Repository Test Campaign",
    campaign_type="Social Media",
    marketing_channel="Instagram",
    target_audience="Young Adults",
    duration_days=30,
    budget=60000,
    marketing_spend=50000,
    impressions=100000,
    clicks=5000,
    conversions=500,
    engagement_rate=5.0,
    ctr=5.0,
    conversion_rate=10.0,
    campaign_start_date="2026-01-01",
    campaign_end_date="2026-01-30",
)

    db.add(campaign)
    db.commit()
    db.refresh(campaign)

    return campaign


def test_create_prediction():
    db = SessionLocal()

    try:
        campaign = create_test_campaign(db)

        prediction = PredictionRepository.create_prediction(
            db=db,
            campaign_id=campaign.id,
            predicted_roi=1250.50,
            predicted_revenue=None,
            predicted_profit=None,
            roi_category="High",
            model_name="Campaign ROI Random Forest",
            model_version="1.0.0",
            prediction_latency_ms=42.5,
        )

        assert prediction.id is not None
        assert prediction.campaign_id == campaign.id
        assert prediction.predicted_roi == 1250.50
        assert prediction.model_name == "Campaign ROI Random Forest"
        assert prediction.model_version == "1.0.0"

    finally:
        db.rollback()
        db.close()


def test_get_prediction_by_id():
    db = SessionLocal()

    try:
        campaign = create_test_campaign(db)

        created_prediction = PredictionRepository.create_prediction(
            db=db,
            campaign_id=campaign.id,
            predicted_roi=850.25,
            predicted_revenue=None,
            predicted_profit=None,
            roi_category="Medium",
            model_name="Campaign ROI Random Forest",
            model_version="1.0.0",
            prediction_latency_ms=40.0,
        )

        prediction = PredictionRepository.get_prediction_by_id(
            db=db,
            prediction_id=created_prediction.id,
        )

        assert prediction is not None
        assert prediction.id == created_prediction.id

    finally:
        db.rollback()
        db.close()


def test_get_predictions_by_campaign_id():
    db = SessionLocal()

    try:
        campaign = create_test_campaign(db)

        PredictionRepository.create_prediction(
            db=db,
            campaign_id=campaign.id,
            predicted_roi=500.0,
            predicted_revenue=None,
            predicted_profit=None,
            roi_category="Low",
            model_name="Campaign ROI Random Forest",
            model_version="1.0.0",
            prediction_latency_ms=41.0,
        )

        PredictionRepository.create_prediction(
            db=db,
            campaign_id=campaign.id,
            predicted_roi=1500.0,
            predicted_revenue=None,
            predicted_profit=None,
            roi_category="High",
            model_name="Campaign ROI Random Forest",
            model_version="1.0.0",
            prediction_latency_ms=43.0,
        )

        predictions = (
            PredictionRepository.get_predictions_by_campaign_id(
                db=db,
                campaign_id=campaign.id,
            )
        )

        assert len(predictions) == 2

        assert all(
            prediction.campaign_id == campaign.id
            for prediction in predictions
        )

    finally:
        db.rollback()
        db.close()
def test_get_prediction_with_campaign():
    db = SessionLocal()

    try:
        campaign = create_test_campaign(db)

        created_prediction = PredictionRepository.create_prediction(
            db=db,
            campaign_id=campaign.id,
            predicted_roi=1750.75,
            predicted_revenue=None,
            predicted_profit=None,
            roi_category="High",
            model_name="Campaign ROI Random Forest",
            model_version="1.0.0",
            prediction_latency_ms=45.0,
        )

        prediction = (
            PredictionRepository.get_prediction_with_campaign(
                db=db,
                prediction_id=created_prediction.id,
            )
        )

        assert prediction is not None
        assert prediction.id == created_prediction.id
        assert prediction.campaign is not None
        assert prediction.campaign.id == campaign.id
        assert prediction.campaign.campaign_name == "Repository Test Campaign"

    finally:
        db.rollback()
        db.close()
