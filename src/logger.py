import logging
import os
from datetime import datetime

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOGGING_FILE_PATH = os.path.join(logs_dir, f"{timestamp}.log")

logging.basicConfig(filename=LOGGING_FILE_PATH,
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )