import logging
import os
from app.core.config import settings

# Dictionary to keep track of configured loggers
_configured_loggers = {}

def setup_logging():
    """
    Configure application logging
    
    This function sets up all loggers used in the application with appropriate
    levels and handlers to ensure clean, readable logs in both development and
    production environments.
    """
    # Get the log level from settings - default to INFO if not specified
    root_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Configure specific loggers with different levels
    main_logger = logging.getLogger("app")
    # Set app logger to DEBUG to see all app module logs
    main_logger.setLevel(logging.DEBUG)
    _configured_loggers["app"] = main_logger

    # Configure specific app modules
    exercises_logger = logging.getLogger("app.api.routes.exercises")
    exercises_logger.setLevel(logging.DEBUG)
    _configured_loggers["app.api.routes.exercises"] = exercises_logger

    # Create and configure API request logger (used in RequestLoggingMiddleware)
    api_logger = logging.getLogger("api")
    api_logger.setLevel(logging.INFO)  # Always set to INFO to ensure logs are shown
    api_logger.propagate = True  # Make sure logs propagate to root logger
    _configured_loggers["api"] = api_logger

    # Create and configure database logger
    db_logger = logging.getLogger("database")
    db_logger.setLevel(logging.INFO)  # Set to INFO normally
    db_logger.propagate = True
    _configured_loggers["database"] = db_logger

    # Configure pymongo logger to reduce noise
    pymongo_logger = logging.getLogger("pymongo")
    pymongo_logger.setLevel(logging.WARNING)  # Only show warnings and errors
    pymongo_logger.propagate = False  # Don't propagate to avoid duplicates

    # Configure motor logger to reduce noise (MongoDB async driver)
    motor_logger = logging.getLogger("motor")
    motor_logger.setLevel(logging.WARNING)  # Only show warnings and errors
    motor_logger.propagate = False  # Don't propagate

    # Reduce noise from uvicorn access logs (which duplicate our custom logs)
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.setLevel(logging.WARNING)
    uvicorn_logger.propagate = False
    
    # Reduce noise from python_multipart
    multipart_logger = logging.getLogger("python_multipart")
    multipart_logger.setLevel(logging.WARNING)
    multipart_logger.propagate = False
    
    # Reduce noise from urllib3
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.WARNING)
    urllib3_logger.propagate = False

    # Log to file if enabled in settings
    if hasattr(settings, 'LOG_TO_FILE') and settings.LOG_TO_FILE:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler('logs/api.log')
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        
        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
        
        # Also add to API logger specifically
        api_logger.addHandler(file_handler)

    print(f"Logging configured. Root level: {logging.getLevelName(root_level)}, App level: {logging.getLevelName(main_logger.level)}")
    return _configured_loggers

def get_logger(name):
    """
    Get a configured logger by name
    
    Args:
        name: The name of the logger to get
        
    Returns:
        A configured logger instance
    """
    # If we've already configured this logger, return it
    if name in _configured_loggers:
        return _configured_loggers[name]
    
    # For app modules, always set to DEBUG level
    if name.startswith("app."):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        _configured_loggers[name] = logger
        return logger
    
    # Check if parent logger is configured
    parts = name.split('.')
    for i in range(len(parts), 0, -1):
        parent_name = '.'.join(parts[:i])
        if parent_name in _configured_loggers:
            logger = logging.getLogger(name)
            # Store in configured loggers
            _configured_loggers[name] = logger
            return logger
    
    # Just return a standard logger if not found in configured loggers
    logger = logging.getLogger(name)
    _configured_loggers[name] = logger
    return logger