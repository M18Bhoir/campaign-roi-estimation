import logging
import os
import sys


LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging():
    """
    Configure application-wide logging.

    Logging level is controlled through the LOG_LEVEL
    environment variable.

    Supported levels:
    DEBUG
    INFO
    WARNING
    ERROR
    CRITICAL
    """

    log_level_name = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).strip().upper()

    valid_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if log_level_name not in valid_levels:
        log_level_name = "INFO"

    log_level = getattr(
        logging,
        log_level_name,
    )

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the requested module.
    """

    return logging.getLogger(name)