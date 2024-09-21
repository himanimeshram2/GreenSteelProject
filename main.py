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

from models.energy_model import GreenSteelModel
from IO.command_line_utils import parse_command_line_args
from IO.json_utilities import load_json_file
from IO.csv_utilities import load_csv_file  # Assuming you have similar functionality
from IO.hdf5_utilities import load_hdf5_data  # Adjust according to actual function signatures
import sys
import time

def main():
    try:
        # Parse command line arguments
        rv = parse_command_line_args()  # Parse args from command line
    except Exception as e:
        print(f"Error parsing command line arguments: {e}")
        return 1  # Return non-zero error code to indicate failure

    try:
        # Load the model parameters based on the file type specified
        if rv.type == 'json':
            data = load_json_file(rv.input)
        elif rv.type == 'csv':
            data = load_csv_file(rv.input)  # Assumes you have a CSV loader implemented
        elif rv.type == 'hdf5':
            data = load_hdf5_data(rv.input, 'dataset_name')  # Modify based on your HDF5 structure
        else:
            print(f"Unsupported file type: {rv.type}")
            return 1

        # Initialize and run the model with loaded data
        theGreenSteelModel = GreenSteelModel(data)  # Assuming your model class can take the data
        theGreenSteelModel.Initialize()
        theGreenSteelModel.Run()
    except FileNotFoundError:
        print(f"Error: The file {rv.input} was not found.")
        return 1
    except Exception as e:
        print(f"Error loading model parameters: {e}")
        return 1

    return 0  # Return zero to indicate success

if __name__ == '__main__':
    start_time = time.time()
    status = main()
    end_time = time.time()
    print(f"Execution completed in {end_time - start_time:.2f} seconds with status code {status}.")
    sys.exit(status)
