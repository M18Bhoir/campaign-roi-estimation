from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.models import ModelMetadata


class ModelMetadataRepository:
    """
    Repository responsible for database operations
    related to ML model metadata.
    """

    # ========================================================
    # CREATE MODEL METADATA
    # ========================================================

    @staticmethod
    def create_model_metadata(
        db: Session,
        model_name: str,
        version: str,
        algorithm: str,
        test_r2: float,
        cv_r2: float,
        mae: float,
        rmse: float,
        training_samples: int,
        feature_count: int,
        trained_at,
        is_active: bool = False,
    ) -> ModelMetadata:
        """
        Create and persist ML model metadata.
        """

        model_metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            algorithm=algorithm,
            test_r2=test_r2,
            cv_r2=cv_r2,
            mae=mae,
            rmse=rmse,
            training_samples=training_samples,
            feature_count=feature_count,
            trained_at=trained_at,
            is_active=is_active,
        )

        db.add(model_metadata)
        db.flush()

        return model_metadata

    # ========================================================
    # GET MODEL METADATA BY ID
    # ========================================================

    @staticmethod
    def get_model_metadata_by_id(
        db: Session,
        model_metadata_id: UUID,
    ) -> ModelMetadata | None:
        """
        Retrieve model metadata using its UUID.
        """

        return (
            db.query(ModelMetadata)
            .filter(ModelMetadata.id == model_metadata_id)
            .first()
        )

    @staticmethod
    def get_model_metadata_by_name_and_version(
        db: Session,
        model_name: str,
        version: str,
    ) -> ModelMetadata | None:
        return (
            db.query(ModelMetadata)
            .filter(
                ModelMetadata.model_name == model_name,
                ModelMetadata.version == version,
            )
            .first()
        )
    # ========================================================
    # GET ACTIVE MODEL
    # ========================================================

    @staticmethod
    def get_active_model(
        db: Session,
    ) -> ModelMetadata | None:
        """
        Retrieve the currently active ML model.
        """

        return (
            db.query(ModelMetadata)
            .filter(ModelMetadata.is_active.is_(True))
            .order_by(
                ModelMetadata.trained_at.desc(),
                ModelMetadata.id.desc(),
            )
            .first()
        )

    # ========================================================
    # GET ALL MODEL METADATA
    # ========================================================

    @staticmethod
    def get_all_model_metadata(
        db: Session,
    ) -> list[ModelMetadata]:
        """
        Retrieve all stored model metadata records.
        """

        return (
            db.query(ModelMetadata)
            .order_by(
                ModelMetadata.trained_at.desc(),
                ModelMetadata.id.desc(),
            )
            .all()
        )

    # ========================================================
    # SET ACTIVE MODEL
    # ========================================================

    @staticmethod
    def set_active_model(
        db: Session,
        model_metadata_id: UUID,
    ) -> ModelMetadata | None:
        """
        Set one model as active and deactivate
        all other model versions.
        """

        model_metadata = (
            db.query(ModelMetadata)
            .filter(ModelMetadata.id == model_metadata_id)
            .first()
        )

        if model_metadata is None:
            return None

        db.query(ModelMetadata).update(
            {
                ModelMetadata.is_active: False,
            }
        )

        model_metadata.is_active = True

        db.flush()
        db.refresh(model_metadata)

        return model_metadata