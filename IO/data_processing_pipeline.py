from hdf5_utilities import load_hdf5_data
from json_utilities import load_json_file, save_data_to_json_file

try:
    # Load configuration
    config = load_json_file("../data/config.json")

    # Load data
    hdf5_data = load_hdf5_data("../data/some_data.h5", "dataset_name")

    # Process data and save results
    processed_data = process_data(hdf5_data, config)  # Ensure process_data is defined or imported
    save_data_to_json_file(processed_data, "../outputs/processed_results.json")
except Exception as e:
    print(f"An error occurred: {e}")
