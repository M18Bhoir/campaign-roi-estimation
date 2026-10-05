from sqlalchemy.orm import Session

from backend.app.repositories.model_metadata_repository import (
    ModelMetadataRepository,
)
from backend.exceptions import ModelNotReadyError


class ModelMetadataService:
    """
    Service responsible for retrieving ML model metadata.
    """

    @staticmethod
    def get_active_model_metadata(
        db: Session,
    ):
        """
        Retrieve metadata for the currently active ML model.
        """

        model_metadata = (
            ModelMetadataRepository.get_active_model(db)
        )

        if model_metadata is None:
            raise ModelNotReadyError(
                "No active ML model metadata is available."
            )

        return model_metadata