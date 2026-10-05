# ============================================================
# ROI CATEGORY
# ============================================================


def get_roi_category(predicted_roi: float) -> str:
    """
    Convert predicted ROI into a business-friendly category.

    Categories:
        < 0       -> Negative
        0 - 99.99 -> Low
        100 - 299.99 -> Moderate
        300 - 499.99 -> High
        >= 500    -> Very High
    """

    roi = float(predicted_roi)

    if roi < 0:
        return "Negative"

    if roi < 100:
        return "Low"

    if roi < 300:
        return "Moderate"

    if roi < 500:
        return "High"

    return "Very High"