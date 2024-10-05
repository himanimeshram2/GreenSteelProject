#!/usr/bin/env python
"""
Copyright (C) 2024, Himani Meshram

This software, known as [ProjectName], is released under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The project uses third party components which may have different licenses. 
Please refer to individual components for more details.

@author: Himani Meshram
"""

import logging
import sys
import time
from models.energy_model import GreenSteelModel
from IO.command_line_utils import parse_command_line_args
from IO.json_utilities import load_json_file
from IO.csv_utilities import load_csv_file
from IO.hdf5_utilities import load_hdf5_data
from visualize_results import plot_levelized_cost, plot_installed_capacities, plot_energy_flows

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Parse command line arguments
        rv = parse_command_line_args()  # Parse args from command line
        logging.info("Command line arguments parsed successfully.")
    except Exception as e:
        logging.error(f"Error parsing command line arguments: {e}")
        return 1  # Return non-zero error code to indicate failure

    try:
        # Load the model parameters based on the file type specified
        if rv.type == 'json':
            data = load_json_file(rv.input)
            logging.info(f"JSON file {rv.input} loaded successfully.")
        elif rv.type == 'csv':
            data = load_csv_file(rv.input)
            logging.info(f"CSV file {rv.input} loaded successfully.")
        elif rv.type == 'hdf5':
            data = load_hdf5_data(rv.input, 'dataset_name')  # Modify 'dataset_name' as needed
            logging.info(f"HDF5 file {rv.input} loaded successfully.")
        else:
            logging.error(f"Unsupported file type: {rv.type}")
            return 1

        # Initialize the model with the loaded data
        theGreenSteelModel = GreenSteelModel(data)  # Assuming your model class takes the data
        theGreenSteelModel.Initialize()
        logging.info("GreenSteelModel initialized successfully.")

        # Run cost optimization to adjust capacities dynamically
        logging.info("Optimizing system capacities for minimum cost...")
        theGreenSteelModel.optimize_system()

        # Run the model with the optimized capacities
        logging.info("Running the model simulation...")
        theGreenSteelModel.Run()

        # Save the results after the simulation
        logging.info("Saving results...")
        theGreenSteelModel.save_results()

        # Load visualization data from the output files and generate plots
        logging.info("Loading output data for visualization...")
        try:
            lcos_data = load_json_file('outputs/Output_results.json')
            capacities_data = load_json_file('outputs/Output_results_capacities.json')
            energy_flows_data = load_json_file('outputs/Output_session.json')

            # Plot Levelized Cost of Steel
            if "levelized_cost_AUD_per_tonne" in lcos_data:
                plot_levelized_cost(lcos_data["levelized_cost_AUD_per_tonne"])
            else:
                logging.warning("Levelized cost data not found in Output_results.json")

            # Plot Installed Capacities
            if capacities_data:
                plot_installed_capacities(capacities_data)
            else:
                logging.warning("Installed capacities data not found in Output_results_capacities.json")

            # Plot Energy Flows
            if energy_flows_data:
                plot_energy_flows(energy_flows_data)
            else:
                logging.warning("Energy flows data not found in Output_session.json")

        except FileNotFoundError as e:
            logging.error(f"Error loading output data for visualization: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during visualization: {e}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Value error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Error loading or running the model: {e}")
        return 1

    logging.info("Model execution completed successfully.")
    return 0  # Return zero to indicate success

if __name__ == '__main__':
    start_time = time.time()
    status = main()
    end_time = time.time()
    logging.info(f"Execution completed in {end_time - start_time:.2f} seconds with status code {status}.")
    sys.exit(status)
