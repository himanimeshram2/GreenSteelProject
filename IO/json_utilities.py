import json

def load_json_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None
    except FileNotFoundError:
        print(f"JSON file not found: {filepath}")
        return None

def save_data_to_json_file(data, filepath):
    try:
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)
    except IOError as e:
        print(f"Could not write to {filepath}: {e}")
