"""
This module contains the logger configuration for the project.
"""
import logging
import os
import coloredlogs
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# os.environ['COLOREDLOGS_LOG_FORMAT']='[%(asctime)s] %(message)s'
# logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO')
# coloredlogs.install(level='INFO', logger=logger)

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['COLOREDLOGS_LOG_FORMAT'] = '[%(asctime)s] %(message)s'

# Create a logger and set its level to INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and set its level to INFO
log_file_path = 'log.txt'  # Change this to your desired log file path
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it on both the console handler and file handler
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Install coloredlogs for console output
coloredlogs.install(level='INFO', logger=logger)
