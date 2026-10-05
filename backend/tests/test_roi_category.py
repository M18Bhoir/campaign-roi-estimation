from backend.app.services.roi_category import (
    get_roi_category,
)


def test_negative_roi():
    assert get_roi_category(-10) == "Negative"


def test_low_roi():
    assert get_roi_category(0) == "Low"
    assert get_roi_category(99.99) == "Low"


def test_moderate_roi():
    assert get_roi_category(100) == "Moderate"
    assert get_roi_category(299.99) == "Moderate"


def test_high_roi():
    assert get_roi_category(300) == "High"
    assert get_roi_category(499.99) == "High"


def test_very_high_roi():
    assert get_roi_category(500) == "Very High"
    assert get_roi_category(4505.77) == "Very High"