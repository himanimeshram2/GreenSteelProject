import json
import os
import logging

def load_json_file(filepath):
    if not os.path.exists(filepath):
        logging.error(f"JSON file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {filepath}: {e}")
        return None

def save_data_to_json_file(data, filepath):
    try:
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)
        logging.info(f"Data successfully saved to {filepath}")
    except IOError as e:
        logging.error(f"Could not write to {filepath}: {e}")
