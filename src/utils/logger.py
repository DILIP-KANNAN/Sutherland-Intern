import logging
import os
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logger(log_file_path: str = None) -> logging.Logger:
    """
    Sets up a logger with standard console output and structured JSON file output.
    """
    logger = logging.getLogger("pattern_miner")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Console handler with clean human-readable formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with structured JSON formatting
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
        
    return logger
