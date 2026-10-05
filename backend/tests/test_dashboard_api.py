from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_dashboard_summary():
    response = client.get(
        "/api/dashboard/summary"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "data" in data
    assert "request_id" in data
    assert "timestamp" in data

    dashboard = data["data"]

    assert "total_predictions" in dashboard
    assert "average_predicted_roi" in dashboard
    assert "highest_predicted_roi" in dashboard
    assert "average_budget" in dashboard
    assert "total_campaigns" in dashboard


def test_dashboard_roi_distribution():
    response = client.get(
        "/api/dashboard/roi-distribution"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert isinstance(data["data"], list)

    for item in data["data"]:
        assert "roi_category" in item
        assert "prediction_count" in item

        assert isinstance(
            item["roi_category"],
            str,
        )

        assert isinstance(
            item["prediction_count"],
            int,
        )


def test_dashboard_channel_performance():
    response = client.get(
        "/api/dashboard/channel-performance"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert isinstance(data["data"], list)

    for item in data["data"]:
        assert "marketing_channel" in item
        assert "prediction_count" in item
        assert "average_predicted_roi" in item

        assert isinstance(
            item["marketing_channel"],
            str,
        )

        assert isinstance(
            item["prediction_count"],
            int,
        )


def test_dashboard_campaign_type_performance():
    response = client.get(
        "/api/dashboard/campaign-type-performance"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert isinstance(data["data"], list)

    for item in data["data"]:
        assert "campaign_type" in item
        assert "prediction_count" in item
        assert "average_predicted_roi" in item

        assert isinstance(
            item["campaign_type"],
            str,
        )

        assert isinstance(
            item["prediction_count"],
            int,
        )


def test_dashboard_roi_trend():
    response = client.get(
        "/api/dashboard/roi-trend"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert isinstance(data["data"], list)

    for item in data["data"]:
        assert "date" in item
        assert "prediction_count" in item
        assert "average_predicted_roi" in item

        assert isinstance(
            item["date"],
            str,
        )

        assert isinstance(
            item["prediction_count"],
            int,
        )


def test_dashboard_recent_predictions():
    response = client.get(
        "/api/dashboard/recent-predictions"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert isinstance(data["data"], list)

    assert len(data["data"]) <= 10

    for item in data["data"]:
        assert "prediction_id" in item
        assert "campaign_id" in item
        assert "campaign_name" in item
        assert "campaign_type" in item
        assert "marketing_channel" in item
        assert "budget" in item
        assert "predicted_roi" in item
        assert "model_version" in item
        assert "created_at" in item


def test_dashboard_recent_predictions_limit():
    response = client.get(
        "/api/dashboard/recent-predictions?limit=5"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) <= 5


def test_dashboard_recent_predictions_limit_above_maximum():
    response = client.get(
        "/api/dashboard/recent-predictions?limit=51"
    )

    assert response.status_code == 422


def test_dashboard_recent_predictions_limit_zero():
    response = client.get(
        "/api/dashboard/recent-predictions?limit=0"
    )

    assert response.status_code == 422