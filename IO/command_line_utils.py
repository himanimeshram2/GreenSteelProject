import argparse

def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Run the GreenSteelModel with various input types")
    parser.add_argument('-i', '--input', required=True, help='Input file with model parameters')
    parser.add_argument('-t', '--type', choices=['json', 'csv', 'hdf5'], default='json', help='Type of the input file (json, csv, or hdf5)')
    parser.add_argument('-o', '--output', help='Output file to save results (optional)')
    return parser.parse_args()
