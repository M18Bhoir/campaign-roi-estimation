"""
Generate realistic synthetic campaign dataset for model training.
Uses real-world benchmark distributions from:
  - Google Ads Industry Benchmarks (WordStream, 2023)
  - HubSpot State of Marketing Report (2023)
  - Nielsen Marketing ROI Report (2023)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N_SAMPLES = 10_000


def generate_dataset() -> pd.DataFrame:
    channels = ["paid_search", "social_media", "email", "display_ads", "influencer", "seo"]
    industries = ["ecommerce", "saas", "retail", "finance", "healthcare", "education"]

    channel = np.random.choice(channels, N_SAMPLES, p=[0.30, 0.28, 0.18, 0.12, 0.07, 0.05])
    industry = np.random.choice(industries, N_SAMPLES, p=[0.28, 0.22, 0.20, 0.12, 0.10, 0.08])

    budget = np.random.lognormal(mean=10.5, sigma=1.2, size=N_SAMPLES).clip(500, 5_000_000)
    duration_days = np.random.choice(range(7, 181), N_SAMPLES)
    audience_size = np.random.lognormal(12.5, 1.5, N_SAMPLES).astype(int).clip(1000, 50_000_000)

    # CTR by channel — based on WordStream benchmarks
    ctr_means = {"paid_search": 0.035, "social_media": 0.009, "email": 0.021,
                 "display_ads": 0.005, "influencer": 0.032, "seo": 0.028}
    ctr = np.array([
        np.random.beta(a=ctr_means[c]*200, b=(1-ctr_means[c])*200)
        for c in channel
    ]).clip(0.001, 0.25)

    cvr = (ctr * np.random.uniform(0.3, 0.9, N_SAMPLES)).clip(0.001, 0.15)

    aov_by_industry = {"ecommerce": 85, "saas": 320, "retail": 55,
                       "finance": 450, "healthcare": 280, "education": 180}
    aov = np.array([
        np.random.lognormal(np.log(aov_by_industry[i]), 0.4)
        for i in industry
    ]).clip(10, 5000)

    seasonality = np.random.uniform(0.7, 2.0, N_SAMPLES)
    competitor_score = np.random.beta(2, 2, N_SAMPLES)

    # Simulate ROI with realistic noise
    # Formula: base ROI driven by marketing math + channel/industry modifiers
    cpc_estimates = {"paid_search": 2.5, "social_media": 0.8, "email": 0.1,
                     "display_ads": 0.5, "influencer": 1.5, "seo": 0.05}
    estimated_clicks = np.array([budget[i] / cpc_estimates[channel[i]] for i in range(N_SAMPLES)])
    estimated_revenue = estimated_clicks * cvr * aov * seasonality
    roi = ((estimated_revenue - budget) / budget * 100
           * np.random.normal(1.0, 0.15, N_SAMPLES)
           * (1 - competitor_score * 0.2))

    df = pd.DataFrame({
        "budget": budget.round(2),
        "channel": channel,
        "industry": industry,
        "duration_days": duration_days,
        "target_audience_size": audience_size,
        "historical_ctr": ctr.round(4),
        "historical_conversion_rate": cvr.round(4),
        "avg_order_value": aov.round(2),
        "seasonality_index": seasonality.round(3),
        "competitor_activity_score": competitor_score.round(3),
        "roi_percent": roi.round(2)  # Target variable
    })

    return df


if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/raw/campaign_data.csv", index=False)
    print(f"Generated {len(df):,} samples → data/raw/campaign_data.csv")
    print(f"\nROI Statistics:")
    print(df["roi_percent"].describe().round(2))
    print(f"\nChannel distribution:\n{df['channel'].value_counts()}")
