import csv
import logging

def load_csv_file(filepath):
    """Load data from a CSV file."""
    data = []
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        logging.info(f"CSV file {filepath} loaded successfully.")
        return data
    except csv.Error as e:
        logging.error(f"Error reading CSV file {filepath}: {e}")
        return None
    except IOError as e:
        logging.error(f"Error opening file {filepath}: {e}")
        return None

def save_data_to_csv_file(data, filepath):
    """Save data to a CSV file."""
    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        logging.info(f"Data successfully saved to {filepath}.")
    except IOError as e:
        logging.error(f"Error saving to CSV file {filepath}: {e}")
