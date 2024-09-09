import logging
import json
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_config(config):
    required_keys = ['name', 'api_key', 'model', 'prompt']
    for model_config in config['models']:
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"Missing required key '{key}' in model configuration")

def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        
        # Load environment variables
        load_dotenv()
        
        # Replace API key placeholders with actual values from environment variables
        for model_config in config['models']:
            model_config['api_key'] = os.getenv(model_config['api_key'])
            if not model_config['api_key']:
                logger.error(f"API key not found for {model_config['name']}")
        
        validate_config(config)
        
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_file}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file: {config_file}")
        raise