import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return str(log_data)

def setup_logging(level: str = "INFO") -> None:
    """Set up structured logging for the application."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Set up specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

def log_model_loading(model_name: str, success: bool, details: Optional[str] = None) -> None:
    """Log model loading events."""
    logger = get_logger("model")
    if success:
        logger.info(f"Model {model_name} loaded successfully", extra={"extra_fields": {"model": model_name, "status": "success"}})
    else:
        logger.error(f"Failed to load model {model_name}", extra={"extra_fields": {"model": model_name, "status": "failed", "details": details}})

def log_inference(model_name: str, inference_time: float, input_shape: tuple, output_shape: tuple) -> None:
    """Log inference events."""
    logger = get_logger("inference")
    logger.info(f"Inference completed for {model_name}", extra={
        "extra_fields": {
            "model": model_name,
            "inference_time_ms": round(inference_time * 1000, 2),
            "input_shape": str(input_shape),
            "output_shape": str(output_shape)
        }
    })

def log_detections(model_name: str, num_detections: int, detections: list) -> None:
    """Log detection results."""
    logger = get_logger("detection")
    logger.info(f"Detection completed for {model_name}", extra={
        "extra_fields": {
            "model": model_name,
            "num_detections": num_detections,
            "detections": detections
        }
    })