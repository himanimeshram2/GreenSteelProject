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
from Models.GreenSteelModel import GreenSteelModel
from IO.CommandLine import ParseCommandLineArgs

def main():
    try:
        # Parse command line arguments
        rv = ParseCommandLineArgs()
    except Exception as e:
        print(f"Error parsing command line arguments: {e}")
        return 1  # Return non-zero error code to indicate failure

    try:
        # Load the model parameters from the specified JSON file
        theGreenSteelModel = GreenSteelModel.FromJsonFile(rv.input)
    except FileNotFoundError:
        print(f"Error: The file {rv.input} was not found.")
        return 1
    except Exception as e:
        print(f"Error loading model parameters: {e}")
        return 1

    # Initialize and run the model
    theGreenSteelModel.Initialize()
    theGreenSteelModel.Run()

    return 0  # Return zero to indicate success

if __name__ == '__main__':
    # Optionally measure execution time
    import time
    start_time = time.time()
    status = main()
    end_time = time.time()
    print(f"Execution completed in {end_time - start_time} seconds with status code {status}.")


 
