import json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.database import SessionLocal
from backend.app.repositories.model_metadata_repository import (
    ModelMetadataRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "model_metadata.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "campaign_roi_model.joblib"
)


def seed_model_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Model metadata file not found: {METADATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_FILE}"
        )

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_name = metadata["model_name"]
    version = metadata["model_version"]

    db = SessionLocal()

    try:
        existing = (
        ModelMetadataRepository.get_model_metadata_by_name_and_version(
            db=db,
            model_name=model_name,
            version=version,
        )
    )

        if existing is not None:
            print(
                f"Model metadata already exists: "
                f"{model_name} v{version}"
            )
            return existing

        artifact_timestamp = datetime.fromtimestamp(
            MODEL_FILE.stat().st_mtime,
            tz=timezone.utc,
        )

        model_metadata = (
            ModelMetadataRepository.create_model_metadata(
                db=db,
                model_name=model_name,
                version=version,
                algorithm=metadata["algorithm"],
                test_r2=metadata["test_r2"],
                cv_r2=metadata["cv_r2"],
                mae=metadata["mae"],
                rmse=metadata["rmse"],
                training_samples=metadata["training_samples"],
                feature_count=metadata["feature_count"],
                trained_at=artifact_timestamp,
                is_active=True,
            )
        )

        db.commit()
        db.refresh(model_metadata)

        print("Model metadata seeded successfully.")
        print(f"Model: {model_metadata.model_name}")
        print(f"Version: {model_metadata.version}")
        print(f"Active: {model_metadata.is_active}")

        return model_metadata

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_model_metadata()
