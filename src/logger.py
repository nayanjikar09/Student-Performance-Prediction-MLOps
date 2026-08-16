import logging
import os
from datetime import datetime

# Create log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs directory path
logs_path = os.path.join(os.getcwd(), "logs")

# Create the directory if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Optional: Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
))

# Get the root logger and add console handler
logger = logging.getLogger()
logger.addHandler(console_handler)

if __name__ == "__main__":
    logging.info("Logging is working!")
    logging.warning("This is a warning message")
    logging.error("This is an error message")