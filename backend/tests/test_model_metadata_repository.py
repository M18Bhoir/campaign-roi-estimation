from datetime import datetime, timezone

from backend.app.database import SessionLocal
from backend.app.models import ModelMetadata
from backend.app.repositories.model_metadata_repository import (
    ModelMetadataRepository,
)


def create_test_model_metadata(
    db,
    version="test-1.0.0",
    is_active=False,
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


def test_create_model_metadata():
    db = SessionLocal()

    try:
        model_metadata = create_test_model_metadata(db)

        assert model_metadata.id is not None
        assert model_metadata.model_name == "Campaign ROI Random Forest"
        assert model_metadata.version == "test-1.0.0"
        assert model_metadata.algorithm == "RandomForestRegressor"
        assert model_metadata.test_r2 == 0.3027
        assert model_metadata.cv_r2 == 0.2872
        assert model_metadata.training_samples == 120000
        assert model_metadata.feature_count == 16
        assert model_metadata.is_active is False

    finally:
        db.rollback()
        db.close()


def test_get_model_metadata_by_id():
    db = SessionLocal()

    try:
        created = create_test_model_metadata(db)

        result = ModelMetadataRepository.get_model_metadata_by_id(
            db=db,
            model_metadata_id=created.id,
        )

        assert result is not None
        assert result.id == created.id
        assert result.version == "test-1.0.0"

    finally:
        db.rollback()
        db.close()


def test_get_active_model():
    db = SessionLocal()

    try:
        created = create_test_model_metadata(
            db,
            version="test-1.0.0",
            is_active=True,
        )

        result = ModelMetadataRepository.get_active_model(db)

        assert result is not None
        assert result.id == created.id
        assert result.is_active is True

    finally:
        db.rollback()
        db.close()


def test_get_all_model_metadata():
    db = SessionLocal()

    try:
        first = create_test_model_metadata(
            db,
            version="test-1.0.0",
        )

        second = create_test_model_metadata(
            db,
            version="2.0.0",
        )

        result = ModelMetadataRepository.get_all_model_metadata(db)

        result_ids = [item.id for item in result]

        assert first.id in result_ids
        assert second.id in result_ids
        assert len(result) >= 2

    finally:
        db.rollback()
        db.close()


def test_set_active_model():
    db = SessionLocal()

    try:
        first = create_test_model_metadata(
            db,
            version="test-1.0.0",
            is_active=True,
        )

        second = create_test_model_metadata(
            db,
            version="2.0.0",
            is_active=False,
        )

        result = ModelMetadataRepository.set_active_model(
            db=db,
            model_metadata_id=second.id,
        )

        assert result is not None
        assert result.id == second.id
        assert result.is_active is True

        db.refresh(first)

        assert first.is_active is False

    finally:
        db.rollback()
        db.close()

def test_get_model_metadata_by_name_and_version():
    db = SessionLocal()

    try:
        created_model = create_test_model_metadata(
            db=db,
            version="test-1.0.0",
            is_active=True,
        )

        found_model = (
            ModelMetadataRepository
            .get_model_metadata_by_name_and_version(
                db=db,
                model_name="Campaign ROI Random Forest",
                version="test-1.0.0",
            )
        )

        assert found_model is not None
        assert found_model.id == created_model.id
        assert found_model.model_name == "Campaign ROI Random Forest"
        assert found_model.version == "test-1.0.0"
        assert found_model.is_active is True

    finally:
        db.rollback()
        db.close()