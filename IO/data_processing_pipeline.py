import logging
from hdf5_utilities import load_hdf5_data
from json_utilities import load_json_file, save_data_to_json_file

def load_and_process_data(config_path, hdf5_path, dataset_name, output_path):
    try:
        # Load configuration
        config = load_json_file(config_path)
        if not config:
            logging.error("Configuration file could not be loaded.")
            return

        # Load data
        hdf5_data = load_hdf5_data(hdf5_path, dataset_name)
        if hdf5_data is None:
            logging.error("HDF5 data could not be loaded.")
            return

        # Process data (ensure process_data is properly defined or imported)
        processed_data = process_data(hdf5_data, config)
        save_data_to_json_file(processed_data, output_path)
        logging.info(f"Data processed and saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
