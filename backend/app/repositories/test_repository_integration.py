from datetime import datetime, timezone

from backend.app.database import SessionLocal
from backend.app.models import Campaign
from backend.app.repositories.campaign_repository import (
    CampaignRepository,
)
from backend.app.repositories.model_metadata_repository import (
    ModelMetadataRepository,
)
from backend.app.repositories.prediction_repository import (
    PredictionRepository,
)


def test_campaign_prediction_integration():
    """
    Verify that a campaign can be created and a prediction
    can be associated with that campaign.
    """

    db = SessionLocal()

    try:
        # ----------------------------------------------------
        # Create Campaign
        # ----------------------------------------------------

        campaign = CampaignRepository.create_campaign(
            db=db,
            campaign_name="Integration Test Campaign",
            campaign_type="Social Media",
            marketing_channel="Instagram",
            target_audience="Young Adults",
            duration_days=30,
            budget=60000.0,
            marketing_spend=50000.0,
            impressions=100000,
            clicks=5000,
            conversions=500,
            engagement_rate=5.0,
            ctr=5.0,
            conversion_rate=10.0,
            campaign_start_date="2026-01-01",
            campaign_end_date="2026-01-30",
        )

        assert campaign.id is not None

        # ----------------------------------------------------
        # Create Prediction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Verify Relationship
        # ----------------------------------------------------

        db.refresh(campaign)

        assert len(campaign.predictions) == 1
        assert campaign.predictions[0].id == prediction.id
        assert campaign.predictions[0].predicted_roi == 1250.50

    finally:
        db.rollback()
        db.close()


def test_multiple_predictions_for_campaign():
    """
    Verify that one campaign can have multiple predictions.
    """

    db = SessionLocal()

    try:
        # ----------------------------------------------------
        # Create Campaign
        # ----------------------------------------------------

        campaign = CampaignRepository.create_campaign(
            db=db,
            campaign_name="Multiple Prediction Test",
            campaign_type="Email",
            marketing_channel="Email",
            target_audience="Existing Customers",
            duration_days=15,
            budget=30000.0,
            marketing_spend=25000.0,
            impressions=50000,
            clicks=3000,
            conversions=300,
            engagement_rate=6.0,
            ctr=6.0,
            conversion_rate=10.0,
            campaign_start_date="2026-02-01",
            campaign_end_date="2026-02-15",
        )

        # ----------------------------------------------------
        # Create First Prediction
        # ----------------------------------------------------

        first_prediction = (
            PredictionRepository.create_prediction(
                db=db,
                campaign_id=campaign.id,
                predicted_roi=500.0,
                predicted_revenue=None,
                predicted_profit=None,
                roi_category="Low",
                model_name="Campaign ROI Random Forest",
                model_version="1.0.0",
                prediction_latency_ms=40.0,
            )
        )

        # ----------------------------------------------------
        # Create Second Prediction
        # ----------------------------------------------------

        second_prediction = (
            PredictionRepository.create_prediction(
                db=db,
                campaign_id=campaign.id,
                predicted_roi=1500.0,
                predicted_revenue=None,
                predicted_profit=None,
                roi_category="High",
                model_name="Campaign ROI Random Forest",
                model_version="1.0.0",
                prediction_latency_ms=41.0,
            )
        )

        assert first_prediction.campaign_id == campaign.id
        assert second_prediction.campaign_id == campaign.id

        predictions = (
            PredictionRepository.get_predictions_by_campaign_id(
                db=db,
                campaign_id=campaign.id,
            )
        )

        assert len(predictions) == 2

    finally:
        db.rollback()
        db.close()


def test_model_metadata_and_prediction_integration():
    """
    Verify that model metadata can be stored and the same
    model version can be associated with a prediction.
    """

    db = SessionLocal()

    try:
        # ----------------------------------------------------
        # Create Model Metadata
        # ----------------------------------------------------

        model_metadata = (
            ModelMetadataRepository.create_model_metadata(
                db=db,
                model_name="Campaign ROI Random Forest",
                version="1.0.0",
                algorithm="RandomForestRegressor",
                test_r2=0.3027,
                cv_r2=0.2872,
                mae=2313.3110,
                rmse=6240.2735,
                training_samples=120000,
                feature_count=16,
                trained_at=datetime.now(timezone.utc),
                is_active=True,
            )
        )

        assert model_metadata.id is not None
        assert model_metadata.is_active is True

        # ----------------------------------------------------
        # Create Campaign
        # ----------------------------------------------------

        campaign = CampaignRepository.create_campaign(
            db=db,
            campaign_name="Model Integration Test",
            campaign_type="Search",
            marketing_channel="Google Ads",
            target_audience="Online Shoppers",
            duration_days=20,
            budget=50000.0,
            marketing_spend=40000.0,
            impressions=80000,
            clicks=4000,
            conversions=400,
            engagement_rate=5.0,
            ctr=5.0,
            conversion_rate=10.0,
            campaign_start_date="2026-03-01",
            campaign_end_date="2026-03-20",
        )

        # ----------------------------------------------------
        # Create Prediction Using Model Version
        # ----------------------------------------------------

        prediction = PredictionRepository.create_prediction(
            db=db,
            campaign_id=campaign.id,
            predicted_roi=900.75,
            predicted_revenue=None,
            predicted_profit=None,
            roi_category="Medium",
            model_name=model_metadata.model_name,
            model_version=model_metadata.version,
            prediction_latency_ms=43.0,
        )

        assert prediction.model_name == model_metadata.model_name
        assert prediction.model_version == model_metadata.version

    finally:
        db.rollback()
        db.close()


def test_campaign_delete_cascades_predictions():
    """
    Verify that deleting a campaign also deletes its
    associated predictions because of ON DELETE CASCADE.
    """

    db = SessionLocal()

    try:
        # ----------------------------------------------------
        # Create Campaign
        # ----------------------------------------------------

        campaign = CampaignRepository.create_campaign(
            db=db,
            campaign_name="Cascade Test Campaign",
            campaign_type="Display",
            marketing_channel="Google Display",
            target_audience="General",
            duration_days=10,
            budget=12000.0,
            marketing_spend=10000.0,
            impressions=20000,
            clicks=1000,
            conversions=100,
            engagement_rate=5.0,
            ctr=5.0,
            conversion_rate=10.0,
            campaign_start_date="2026-04-01",
            campaign_end_date="2026-04-10",
        )

        # ----------------------------------------------------
        # Create Prediction
        # ----------------------------------------------------

        prediction = PredictionRepository.create_prediction(
            db=db,
            campaign_id=campaign.id,
            predicted_roi=700.0,
            predicted_revenue=None,
            predicted_profit=None,
            roi_category="Medium",
            model_name="Campaign ROI Random Forest",
            model_version="1.0.0",
            prediction_latency_ms=44.0,
        )

        prediction_id = prediction.id
        campaign_id = campaign.id

        # ----------------------------------------------------
        # Delete Campaign
        # ----------------------------------------------------

        db.delete(campaign)
        db.commit()

        # ----------------------------------------------------
        # Verify Campaign Deleted
        # ----------------------------------------------------

        deleted_campaign = (
            db.query(Campaign)
            .filter(Campaign.id == campaign_id)
            .first()
        )

        assert deleted_campaign is None

        # ----------------------------------------------------
        # Verify Prediction Deleted
        # ----------------------------------------------------

        deleted_prediction = (
            PredictionRepository.get_prediction_by_id(
                db=db,
                prediction_id=prediction_id,
            )
        )

        assert deleted_prediction is None

    finally:
        db.rollback()
        db.close()