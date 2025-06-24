# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
import logging
import datetime
import sys

import os


def setup_logger(name="suvetha", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create file handler
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%H_%M_%d-%m-%Y")
        file_handler = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
    
        file_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def log_info(logger, message):
    logger.info(message)