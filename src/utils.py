import logging
import sys

def get_logger(name: str = "ml_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
