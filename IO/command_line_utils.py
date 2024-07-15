import argparse

def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Run the GreenSteelModel")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input JSON file with model parameters')
    args = parser.parse_args()
    return args


