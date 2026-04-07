"""
Project Logging Utility Class
Implemented based on loguru, supports .env configuration for dual output to console/file, and automatically generates logs/app_YYYYMMDD.log
Features:
1. Configuration-driven: Output and log level can be modified via .env switches
2. Automatic Path: File logs are output by default to project_root/logs/app_YYYYMMDD.log
3. Automatic Cleanup: Retains logs as per configuration and automatically deletes expired files
4. UTF-8 Friendly: Ensures no Chinese character encoding issues
5. Asynchronous Safety: Enables asynchronous queuing, supports multi-threading/asynchronous scenarios, and avoids log disorder
6. Ready to Use: All project modules can directly import and use the logger
7. Precise Location: Displays the actual call location in business modules by penetrating loguru internals and the utility class itself
"""
import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


# -------------------------- Step 1: Load .env configuration file --------------------------
load_dotenv()

# -------------------------- Step 2: Read .env configuration (with default values to prevent missing configurations) --------------------------
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

# -------------------------- Step 3: Define log path (automatically infer project root) --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# -------------------------- Step 4: Define log format (colorful, structured, and readable) --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

# -------------------------- Step 5: Initialize log configuration (core method) --------------------------
def init_logger():
    """
    Initialize global log configuration
    1. Remove loguru's default console output (to avoid duplicate printing)
    2. Enable/disable console output based on .env configuration
    3. Enable/disable file output based on .env configuration (automatically creates logs directory)
    4. Configure log format, level, segmentation, and retention strategy
    :return: Configured loguru logger instance
    """
    # 1. Remove loguru's default console output
    logger.remove()

    # 2. Configure console output (if .env enables)
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True
        )

    # 3. Configure file output (if .env enables)
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    return logger

# -------------------------- Step 6: Initialize and ultimately correct global logger --------------------------
base_logger = init_logger()

def fix_log_position(record):
    """Iterate through the call stack, skip loguru internal frames + the utility class itself, extract actual business code call location"""
    for frame in inspect.stack():
        # Ultimate filtering: Exclude loguru internal + exclude the utility class logger.py itself, directly locate the business module
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # Update the log record with the actual business code location
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break

# Apply ultimate correction, export globally available logger
logger = base_logger.patch(fix_log_position)

# -------------------------- Test code (verify the fix) --------------------------
if __name__ == '__main__':
    logger.info("[Test] logger.py internal call (only for testing, business module calls will display correct file names)")
    print(f"Log file output path: {LOG_FILE_PATH}")