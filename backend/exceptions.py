class PredictionServiceError(Exception):
    """
    Base exception for prediction-service errors.
    """

    pass


class ModelNotReadyError(PredictionServiceError):
    """
    Raised when the ML model is unavailable.
    """

    pass


class PredictionError(PredictionServiceError):
    """
    Raised when prediction generation fails.
    """

    pass