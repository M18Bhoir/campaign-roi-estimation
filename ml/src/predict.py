"""Inference entry points."""
import joblib
import pandas as pd

from config import MODEL_FILE


class ROIPredictor:
    """
    Production ROI prediction service.
    """

    def __init__(self):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_FILE}"
            )

        self.pipeline = joblib.load(
            MODEL_FILE
        )

    def predict(self, campaign_data: dict) -> float:
        """
        Generate ROI prediction from campaign data.
        """

        input_df = pd.DataFrame(
            [campaign_data]
        )

        prediction = self.pipeline.predict(
            input_df
        )

        return float(prediction[0])


if __name__ == "__main__":

    sample_campaign = {
        "Budget": 100000,
        "CompetitorScore": 70,
        "CampaignDurationDays": 30,
        "DiscountPercent": 10,

        "Platform": "Instagram",
        "Region": "Asia Pacific",
        "Device": "Mobile",
        "CustomerSegment": "Returning",
        "ProductCategory": "Electronics",
        "CampaignType": "Sales",
        "Season": "Summer",
        "MarketingObjective": "Sales Growth",
    }

    predictor = ROIPredictor()

    prediction = predictor.predict(
        sample_campaign
    )

    print(
        f"Predicted ROI: {prediction:.4f}"
    )