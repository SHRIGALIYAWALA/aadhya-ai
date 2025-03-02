import logging
import traceback

def setup_logger(name="AadhyaAI", log_file="aadhya.log", level=logging.INFO):
    """Sets up a logger for AadhyaAI."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def log_error(exception):
    """Logs an error with traceback."""
    error_message = f"Exception: {str(exception)}\n{traceback.format_exc()}"
    logger.error(error_message)

def format_data(data):
    """Formats data into a structured string."""
    if isinstance(data, dict):
        return " | ".join([f"{key}: {value}" for key, value in data.items()])
    return str(data)

def safe_execute(func, *args, **kwargs):
    """Executes a function safely, logging any errors."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_error(e)
        return None
