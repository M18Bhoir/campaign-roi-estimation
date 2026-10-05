import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)



VALID_CAMPAIGN = {
    # Application / database fields
    "campaign_name": "Test Campaign",
    "marketing_channel": "Social Media",
    "target_audience": "Young Adults",
    "campaign_start_date": "2025-10-01",
    "campaign_end_date": "2025-12-11",

    # ML features
    "budget": 40365.77,
    "competitor_score": 59,
    "campaign_duration_days": 72,
    "discount_percent": 0.34,
    "platform": "Twitter (X)",
    "region": "Asia Pacific",
    "device": "Smart TV",
    "customer_segment": "New",
    "product_category": "Real Estate",
    "campaign_type": "Sales",
    "season": "Holiday",
    "marketing_objective": "Sales Growth",
    "campaign_year": 2025,
    "campaign_month": 10,
    "campaign_quarter": 4,
    "campaign_day_of_week": 3,
}


# ============================================================
# Basic API Tests
# ============================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["status"] == "running"
    assert "request_id" in data
    assert "timestamp" in data


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["status"] == "healthy"
    assert "model_ready" in data
    assert "request_id" in data
    assert "timestamp" in data


def test_model_info():
    response = client.get("/api/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "data" in data
    assert "request_id" in data
    assert "timestamp" in data


# ============================================================
# Prediction Tests
# ============================================================

def test_valid_prediction():
    response = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "data" in data

    prediction = data["data"]

    assert "campaign_id" in prediction
    assert "prediction_id" in prediction
    assert "predicted_roi" in prediction
    assert "model_name" in prediction
    assert "model_version" in prediction
    assert "prediction_latency_ms" in prediction

    assert isinstance(
        prediction["predicted_roi"],
        (int, float),
    )

    assert prediction["prediction_latency_ms"] >= 0

    assert "request_id" in data
    assert "timestamp" in data


def test_prediction_consistency():
    response_1 = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    response_2 = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    prediction_1 = response_1.json()["data"]["predicted_roi"]
    prediction_2 = response_2.json()["data"]["predicted_roi"]

    assert prediction_1 == pytest.approx(
        prediction_2,
        rel=1e-12,
        abs=1e-12,
    )

def test_predict_rejects_invalid_competitor_score():
    payload = {
        "campaign_name": "Invalid Competitor Score Campaign",
        "marketing_channel": "Digital",
        "target_audience": "Young Adults",
        "campaign_start_date": "2026-10-01",
        "campaign_end_date": "2026-10-10",
        "budget": 50000,
        "competitor_score": 101,
        "campaign_duration_days": 10,
        "discount_percent": 10,
        "platform": "Google",
        "region": "West",
        "device": "Mobile",
        "customer_segment": "Premium",
        "product_category": "Electronics",
        "campaign_type": "Sales",
        "season": "Summer",
        "marketing_objective": "Conversions",
        "campaign_year": 2026,
        "campaign_month": 10,
        "campaign_quarter": 4,
        "campaign_day_of_week": 3,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_negative_competitor_score():
    payload = {
        "campaign_name": "Negative Competitor Score Campaign",
        "marketing_channel": "Digital",
        "target_audience": "Young Adults",
        "campaign_start_date": "2026-10-01",
        "campaign_end_date": "2026-10-10",
        "budget": 50000,
        "competitor_score": -1,
        "campaign_duration_days": 10,
        "discount_percent": 10,
        "platform": "Google",
        "region": "West",
        "device": "Mobile",
        "customer_segment": "Premium",
        "product_category": "Electronics",
        "campaign_type": "Sales",
        "season": "Summer",
        "marketing_objective": "Conversions",
        "campaign_year": 2026,
        "campaign_month": 10,
        "campaign_quarter": 4,
        "campaign_day_of_week": 3,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_discount_above_100():
    payload = {
        "campaign_name": "Invalid Discount Campaign",
        "marketing_channel": "Digital",
        "target_audience": "Young Adults",
        "campaign_start_date": "2026-10-01",
        "campaign_end_date": "2026-10-10",
        "budget": 50000,
        "competitor_score": 50,
        "campaign_duration_days": 10,
        "discount_percent": 101,
        "platform": "Google",
        "region": "West",
        "device": "Mobile",
        "customer_segment": "Premium",
        "product_category": "Electronics",
        "campaign_type": "Sales",
        "season": "Summer",
        "marketing_objective": "Conversions",
        "campaign_year": 2026,
        "campaign_month": 10,
        "campaign_quarter": 4,
        "campaign_day_of_week": 3,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_negative_discount():
    payload = {
        "campaign_name": "Negative Discount Campaign",
        "marketing_channel": "Digital",
        "target_audience": "Young Adults",
        "campaign_start_date": "2026-10-01",
        "campaign_end_date": "2026-10-10",
        "budget": 50000,
        "competitor_score": 50,
        "campaign_duration_days": 10,
        "discount_percent": -1,
        "platform": "Google",
        "region": "West",
        "device": "Mobile",
        "customer_segment": "Premium",
        "product_category": "Electronics",
        "campaign_type": "Sales",
        "season": "Summer",
        "marketing_objective": "Conversions",
        "campaign_year": 2026,
        "campaign_month": 10,
        "campaign_quarter": 4,
        "campaign_day_of_week": 3,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


# ============================================================
# Validation Tests
# ============================================================

def test_negative_budget():
    payload = VALID_CAMPAIGN.copy()
    payload["budget"] = -100

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_zero_campaign_duration():
    payload = VALID_CAMPAIGN.copy()
    payload["campaign_duration_days"] = 0

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_month():
    payload = VALID_CAMPAIGN.copy()
    payload["campaign_month"] = 13

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_quarter():
    payload = VALID_CAMPAIGN.copy()
    payload["campaign_quarter"] = 5

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_day_of_week():
    payload = VALID_CAMPAIGN.copy()
    payload["campaign_day_of_week"] = 7

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_budget():
    payload = VALID_CAMPAIGN.copy()
    del payload["budget"]

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_unexpected_feature():
    payload = VALID_CAMPAIGN.copy()
    payload["ROI"] = 5000

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_budget_type():
    payload = VALID_CAMPAIGN.copy()
    payload["budget"] = "not-a-number"

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_month_type():
    payload = VALID_CAMPAIGN.copy()
    payload["campaign_month"] = "October"

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_empty_platform():
    payload = VALID_CAMPAIGN.copy()
    payload["platform"] = ""

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_null_budget():
    payload = VALID_CAMPAIGN.copy()
    payload["budget"] = None

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# Response Structure Tests
# ============================================================

def test_prediction_response_structure():
    response = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) == {
        "success",
        "data",
        "request_id",
        "timestamp",
    }

    assert set(data["data"].keys()) == {
        "campaign_id",
        "prediction_id",
        "predicted_roi",
        "roi_category",
        "model_name",
        "model_version",
        "prediction_latency_ms",
    }

    assert data["data"]["campaign_id"]
    assert data["data"]["prediction_id"]

    assert isinstance(
        data["data"]["predicted_roi"],
        (int, float),
    )

    assert isinstance(
        data["data"]["model_name"],
        str,
    )

    assert isinstance(
        data["data"]["model_version"],
        str,
    )

    assert isinstance(
        data["data"]["prediction_latency_ms"],
        (int, float),
    )

    assert data["data"]["prediction_latency_ms"] >= 0

    assert isinstance(
    data["data"]["roi_category"],
    str,
)

    assert data["data"]["roi_category"] in {
        "Negative",
        "Low",
        "Moderate",
        "High",
        "Very High",
    }


def test_prediction_is_finite():
    response = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    assert response.status_code == 200

    prediction = response.json()["data"]["predicted_roi"]

    assert prediction == prediction
    assert abs(prediction) != float("inf")


def test_request_id_header():
    response = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    assert response.status_code == 200

    body_request_id = response.json()["request_id"]
    header_request_id = response.headers.get("X-Request-ID")

    assert header_request_id is not None
    assert body_request_id == header_request_id


def test_request_ids_are_unique():
    response_1 = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    response_2 = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    request_id_1 = response_1.json()["request_id"]
    request_id_2 = response_2.json()["request_id"]

    assert request_id_1 != request_id_2


# ============================================================
# Error Response Tests
# ============================================================

def test_validation_error_structure():
    payload = VALID_CAMPAIGN.copy()
    payload["budget"] = -500

    response = client.post(
        "/api/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["success"] is False
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "message" in data["error"]
    assert "details" in data["error"]
    assert "request_id" in data
    assert "timestamp" in data


def test_unknown_endpoint():
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404


# ============================================================
# Prediction Detail API Tests
# ============================================================

def test_prediction_detail_valid_id():
    create_response = client.post(
        "/api/predict",
        json=VALID_CAMPAIGN,
    )

    assert create_response.status_code == 200

    create_data = create_response.json()

    prediction_id = create_data["data"]["prediction_id"]

    response = client.get(
        f"/api/predictions/{prediction_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "data" in data

    prediction = data["data"]

    assert prediction["prediction_id"] == prediction_id

    assert "campaign_id" in prediction
    assert "campaign_name" in prediction
    assert "predicted_roi" in prediction
    assert "roi_category" in prediction
    assert "model_name" in prediction
    assert "model_version" in prediction

def test_prediction_detail_not_found():
    response = client.get(
        "/api/predictions/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404

    data = response.json()

    assert data["success"] is False
    assert data["error"]["code"] == "PREDICTION_NOT_FOUND"


def test_prediction_detail_invalid_uuid():
    response = client.get(
        "/api/predictions/not-a-valid-uuid"
    )

    assert response.status_code == 422


# ============================================================
# Pagination Tests
# ============================================================

def test_prediction_history_default_pagination():
    response = client.get("/api/predictions")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "data" in data
    assert "count" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert "request_id" in data
    assert "timestamp" in data

    assert data["limit"] == 20
    assert data["offset"] == 0
    assert data["count"] <= 20
    assert data["total"] >= 0

    assert isinstance(data["data"], list)


def test_prediction_history_first_page():
    response = client.get(
        "/api/predictions?limit=5&offset=0"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["limit"] == 5
    assert data["offset"] == 0
    assert data["count"] <= 5
    assert data["total"] >= data["count"]

    assert isinstance(data["data"], list)


def test_prediction_history_second_page():
    first_response = client.get(
        "/api/predictions?limit=5&offset=0"
    )

    second_response = client.get(
        "/api/predictions?limit=5&offset=5"
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_data = first_response.json()
    second_data = second_response.json()

    assert first_data["limit"] == 5
    assert first_data["offset"] == 0

    assert second_data["limit"] == 5
    assert second_data["offset"] == 5

    first_ids = {
        item["prediction_id"]
        for item in first_data["data"]
    }

    second_ids = {
        item["prediction_id"]
        for item in second_data["data"]
    }

    assert first_ids.isdisjoint(second_ids)


def test_prediction_history_limit_zero():
    response = client.get(
        "/api/predictions?limit=0"
    )

    assert response.status_code == 422


def test_prediction_history_limit_above_maximum():
    response = client.get(
        "/api/predictions?limit=101"
    )

    assert response.status_code == 422


def test_prediction_history_negative_offset():
    response = client.get(
        "/api/predictions?offset=-1"
    )

    assert response.status_code == 422


# ============================================================
# Prediction History Filter Tests
# ============================================================


def test_prediction_history_filter_by_campaign_type():
    response = client.get(
        "/api/predictions?campaign_type=Sales"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["count"] <= data["total"]

    for item in data["data"]:
        assert item["campaign_type"] == "Sales"


def test_prediction_history_filter_by_marketing_channel():
    response = client.get(
        "/api/predictions?marketing_channel=Social%20Media"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["count"] <= data["total"]

    for item in data["data"]:
        assert item["marketing_channel"] == "Social Media"


def test_prediction_history_filter_by_model_version():
    response = client.get(
        "/api/predictions?model_version=1.0.0"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["count"] <= data["total"]

    for item in data["data"]:
        assert item["model_version"] == "1.0.0"


def test_prediction_history_combined_filters():
    response = client.get(
        "/api/predictions"
        "?campaign_type=Sales"
        "&marketing_channel=Social%20Media"
        "&model_version=1.0.0"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["count"] <= data["total"]

    for item in data["data"]:
        assert item["campaign_type"] == "Sales"
        assert item["marketing_channel"] == "Social Media"
        assert item["model_version"] == "1.0.0"


def test_prediction_history_filters_with_pagination():
    response = client.get(
        "/api/predictions"
        "?campaign_type=Sales"
        "&marketing_channel=Social%20Media"
        "&limit=5"
        "&offset=0"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["limit"] == 5
    assert data["offset"] == 0
    assert data["count"] <= 5
    assert data["total"] >= data["count"]

    for item in data["data"]:
        assert item["campaign_type"] == "Sales"
        assert item["marketing_channel"] == "Social Media"


def test_prediction_history_nonexistent_filter():
    response = client.get(
        "/api/predictions"
        "?campaign_type=NonExistentCampaignType"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["count"] == 0
    assert data["total"] == 0
    assert data["data"] == []

def test_model_metadata():
    response = client.get("/api/model/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "data" in data
    assert "request_id" in data
    assert "timestamp" in data

    model = data["data"]

    assert "model_id" in model
    assert "model_name" in model
    assert "version" in model
    assert "algorithm" in model
    assert "test_r2" in model
    assert "cv_r2" in model
    assert "mae" in model
    assert "rmse" in model
    assert "training_samples" in model
    assert "feature_count" in model
    assert "trained_at" in model
    assert "is_active" in model


def test_model_metadata_values():
    response = client.get("/api/model/metadata")

    assert response.status_code == 200

    model = response.json()["data"]

    assert model["model_name"] == "Campaign ROI Random Forest"
    assert model["version"] == "1.0.0"
    assert model["algorithm"] == "RandomForestRegressor"

    assert model["test_r2"] == 0.302672
    assert model["cv_r2"] == 0.287157

    assert model["training_samples"] == 120000
    assert model["feature_count"] == 16

    assert model["is_active"] is True


def test_model_metadata_request_id_header():
    response = client.get("/api/model/metadata")

    assert response.status_code == 200

    assert "X-Request-ID" in response.headers

    request_id_header = response.headers["X-Request-ID"]
    request_id_body = response.json()["request_id"]

    assert request_id_header == request_id_body