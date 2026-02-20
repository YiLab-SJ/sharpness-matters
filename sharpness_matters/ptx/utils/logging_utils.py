"""
Logging utility functions
"""

import logging
import colorlog


def initialize_logger(level="debug"):
    if level == "debug":
        level = logging.DEBUG
    elif level == "info":
        level = logging.INFO
    elif level == "warning":
        level = logging.WARNING
    elif level == "error":
        level = logging.ERROR
    elif level == "critical":
        level = logging.CRITICAL
    else:
        raise ("Invalid level: Choose one of debug, info, warning, error, critical")
    logger = logging.getLogger("colored_logger")
    logger.setLevel(level)

    # Prevent messages from propagating to ancestor loggers
    logger.propagate = False

    # Only add the handler if it hasn't been added already
    if not logger.hasHandlers():
        handler = logging.StreamHandler()

        # Define the colored formatter, where the message color follows the logging level
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(log_color)s%(message)s%(reset)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
