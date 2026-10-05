from datetime import date

from backend.app.database import SessionLocal
from backend.app.models import Campaign, Prediction
from backend.app.services.prediction_service import PredictionService
from backend.exceptions import PredictionError


def test_predict_and_save_rolls_back_campaign_when_prediction_creation_fails(
    monkeypatch,
):
    db = SessionLocal()

    campaign_name = "Transaction Rollback Test"

    try:
        service = PredictionService()

        prediction_input = {
            "Budget": 50000,
            "CompetitorScore": 50,
            "CampaignDurationDays": 10,
            "DiscountPercent": 10,
            "Platform": "Google",
            "Region": "West",
            "Device": "Mobile",
            "CustomerSegment": "Premium",
            "ProductCategory": "Electronics",
            "CampaignType": "Sales",
            "Season": "Summer",
            "MarketingObjective": "Conversions",
            "CampaignYear": 2026,
            "CampaignMonth": 10,
            "CampaignQuarter": 4,
            "CampaignDayOfWeek": 3,
        }

        campaign_data = {
            "campaign_name": campaign_name,
            "campaign_type": "Sales",
            "marketing_channel": "Google",
            "target_audience": "Young Adults",
            "duration_days": 10,
            "budget": 50000,
            "campaign_start_date": date(2026, 10, 1),
            "campaign_end_date": date(2026, 10, 10),
        }

        def failing_create_prediction(*args, **kwargs):
            raise RuntimeError(
                "Simulated prediction creation failure."
            )

        monkeypatch.setattr(
            "backend.app.services.prediction_service."
            "PredictionRepository.create_prediction",
            failing_create_prediction,
        )

        try:
            service.predict_and_save(
                db=db,
                prediction_input=prediction_input,
                campaign_data=campaign_data,
            )

            assert False, "Expected PredictionError was not raised."

        except PredictionError:
            pass

        # Verify the campaign was rolled back.
        campaign = (
            db.query(Campaign)
            .filter(Campaign.campaign_name == campaign_name)
            .first()
        )

        assert campaign is None

        # Verify no prediction is associated with the rolled-back campaign.
        predictions = (
            db.query(Prediction)
            .join(Campaign, Prediction.campaign_id == Campaign.id)
            .filter(Campaign.campaign_name == campaign_name)
            .all()
        )

        assert predictions == []

    finally:
        db.rollback()
        db.close()