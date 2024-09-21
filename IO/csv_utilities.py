import csv

def load_csv_file(filepath):
    """Load data from a CSV file."""
    data = []
    with open(filepath, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def save_data_to_csv_file(data, filepath):
    """Save data to a CSV file."""
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
