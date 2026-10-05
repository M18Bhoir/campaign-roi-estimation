from datetime import datetime, timezone

from backend.app.database import SessionLocal
from backend.app.repositories.model_metadata_repository import (
    ModelMetadataRepository,
)
from backend.app.services.model_metadata_service import (
    ModelMetadataService,
)
from backend.exceptions import ModelNotReadyError


def create_test_model_metadata(
    db,
    version="test-1.0.0",
    is_active=True,
):
    return ModelMetadataRepository.create_model_metadata(
        db=db,
        model_name="Campaign ROI Random Forest",
        version=version,
        algorithm="RandomForestRegressor",
        test_r2=0.3027,
        cv_r2=0.2872,
        mae=2313.3110,
        rmse=6240.2735,
        training_samples=120000,
        feature_count=16,
        trained_at=datetime.now(timezone.utc),
        is_active=is_active,
    )


def test_get_active_model_metadata():
    db = SessionLocal()

    try:
        created_model = create_test_model_metadata(db)

        result = ModelMetadataService.get_active_model_metadata(db)

        assert result is not None
        assert result.id == created_model.id
        assert result.model_name == "Campaign ROI Random Forest"
        assert result.version == "test-1.0.0"
        assert result.is_active is True

    finally:
        db.rollback()
        db.close()


def test_get_active_model_metadata_raises_when_no_active_model():
    db = SessionLocal()

    try:
        result = ModelMetadataRepository.get_active_model(db)

        if result is not None:
            result.is_active = False
            db.flush()

        try:
            ModelMetadataService.get_active_model_metadata(db)
            assert False, "Expected ModelNotReadyError was not raised."

        except ModelNotReadyError as error:
            assert str(error) == (
                "No active ML model metadata is available."
            )

    finally:
        db.rollback()
        db.close()