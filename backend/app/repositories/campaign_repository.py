from datetime import date

from sqlalchemy.orm import Session

from backend.app.models import Campaign


class CampaignRepository:

    @staticmethod
    def create_campaign(
        db: Session,
        campaign_name: str,
        campaign_type: str,
        marketing_channel: str,
        target_audience: str,
        duration_days: int,
        budget: float,
        marketing_spend: float | None,
        impressions: int,
        clicks: int,
        conversions: int,
        engagement_rate: float | None,
        ctr: float | None,
        conversion_rate: float | None,
        campaign_start_date: date,
        campaign_end_date: date,
    ) -> Campaign:

        campaign = Campaign(
            campaign_name=campaign_name,
            campaign_type=campaign_type,
            marketing_channel=marketing_channel,
            target_audience=target_audience,
            duration_days=duration_days,
            budget=budget,
            marketing_spend=marketing_spend,
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            engagement_rate=engagement_rate,
            ctr=ctr,
            conversion_rate=conversion_rate,
            campaign_start_date=campaign_start_date,
            campaign_end_date=campaign_end_date,
        )

        db.add(campaign)

        # Flush sends INSERT to the database without committing.
        # This gives us campaign.id while keeping the transaction open.
        db.flush()

        return campaign