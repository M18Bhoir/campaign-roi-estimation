from predictor import predictor


# ============================================================
# TEST DATA
# ============================================================

sample_campaign = {
    "Budget": 40365.77,
    "CompetitorScore": 59,
    "CampaignDurationDays": 72,
    "DiscountPercent": 0.34,

    "Platform": "Twitter (X)",
    "Region": "Asia Pacific",
    "Device": "Smart TV",
    "CustomerSegment": "New",
    "ProductCategory": "Real Estate",
    "CampaignType": "Sales",
    "Season": "Holiday",
    "MarketingObjective": "Sales Growth",

    "CampaignYear": 2025,
    "CampaignMonth": 10,
    "CampaignQuarter": 4,
    "CampaignDayOfWeek": 3,
}


# ============================================================
# MODEL STATUS
# ============================================================

print("=" * 70)
print("PRODUCTION PREDICTOR TEST")
print("=" * 70)

print(
    "\nModel ready:",
    predictor.is_ready()
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

model_info = predictor.get_model_info()

for key, value in model_info.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION")
print("=" * 70)

result = predictor.predict(
    sample_campaign
)

print(
    f"\nPredicted ROI: "
    f"{result['predicted_roi']:,.4f}"
)

print(
    f"Model: "
    f"{result['model_name']}"
)

print(
    f"Version: "
    f"{result['model_version']}"
)


# ============================================================
# MISSING FEATURE TEST
# ============================================================

print("\n" + "=" * 70)
print("MISSING FEATURE VALIDATION")
print("=" * 70)

invalid_campaign = sample_campaign.copy()

del invalid_campaign[
    "Budget"
]


try:

    predictor.predict(
        invalid_campaign
    )

    raise RuntimeError(
        "Missing feature was unexpectedly accepted."
    )

except ValueError as error:

    print(
        "Correctly rejected:"
    )

    print(
        error
    )


# ============================================================
# UNEXPECTED FEATURE TEST
# ============================================================

print("\n" + "=" * 70)
print("UNEXPECTED FEATURE VALIDATION")
print("=" * 70)

invalid_campaign = sample_campaign.copy()

invalid_campaign[
    "ROI"
] = 5000


try:

    predictor.predict(
        invalid_campaign
    )

    raise RuntimeError(
        "Unexpected feature was unexpectedly accepted."
    )

except ValueError as error:

    print(
        "Correctly rejected:"
    )

    print(
        error
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("PREDICTOR TEST COMPLETED")
print("=" * 70)