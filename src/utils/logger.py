import logging
import sys
import os
from config.settings import BASE_DIR

def setup_logger(name: str) -> logging.Logger:
    """
    Configura un logger estandarizado para los agentes del sistema.
    Escribe tanto en la consola como en un archivo de log diario.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Archivo
        log_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "system.log"), encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger