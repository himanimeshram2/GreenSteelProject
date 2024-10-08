import json
import matplotlib.pyplot as plt
import numpy as np
from IO.json_utilities import load_json_file

def plot_levelized_cost(lcos, save_path=None):
    """
    Plot the levelized cost of steel in AUD per tonne.
    
    Parameters:
        lcos (float): The levelized cost of steel in AUD per tonne.
        save_path (str): Optional file path to save the plot.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.bar(['Levelized Cost of Steel'], [lcos], color='skyblue')
        plt.ylabel('AUD per Tonne')
        plt.title('Levelized Cost of Steel (AUD/tonne)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Levelized Cost plot saved to {save_path}")
        
        plt.show()
    except Exception as e:
        print(f"Error plotting levelized cost: {e}")

def plot_installed_capacities(capacities, save_path=None):
    """
    Plot the installed capacities for different energy sources and storage.
    
    Parameters:
        capacities (dict): A dictionary with capacity names as keys and values as MW or tonnes.
        save_path (str): Optional file path to save the plot.
    """
    try:
        if not capacities:
            print("No capacities to display.")
            return

        labels = list(capacities.keys())
        values = list(capacities.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='lightgreen')
        plt.ylabel('Capacity (MW or Tonne)')
        plt.title('Installed Capacities (MW/Tonne)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Installed Capacities plot saved to {save_path}")
        
        plt.show()
    except Exception as e:
        print(f"Error plotting installed capacities: {e}")

def plot_energy_flows(energy_flows, save_path=None):
    """
    Plot energy flows over time, e.g., wind and solar generation, battery charge/discharge.
    
    Parameters:
        energy_flows (dict): Dictionary containing hourly energy flow data.
        save_path (str): Optional file path to save the plot.
    """
    try:
        plt.figure(figsize=(12, 6))
        time = np.arange(len(energy_flows["Wind generation (MWh)"]))

        # Plot each energy source or flow
        plt.plot(time, energy_flows["Wind generation (MWh)"], label='Wind Generation', color='blue')
        plt.plot(time, energy_flows["Solar generation (MWh)"], label='Solar Generation', color='orange')
        plt.plot(time, energy_flows["Battery charge (MWh)"], label='Battery Charge', color='green')
        plt.plot(time, energy_flows["Battery discharge (MWh)"], label='Battery Discharge', color='red')
        plt.plot(time, energy_flows["Electrolysis (MWh)"], label='Electrolysis', color='purple')

        plt.xlabel('Time (hours)')
        plt.ylabel('Energy (MWh)')
        plt.title('Hourly Energy Flows Over the Year')
        plt.legend(loc='upper right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Energy Flows plot saved to {save_path}")
        
        plt.show()
    except KeyError as e:
        print(f"Missing data for key: {e}")
    except Exception as e:
        print(f"Error plotting energy flows: {e}")

def main():
    try:
        # Load results from JSON files
        lcos_data = load_json_file('outputs/Output_results.json')
        capacities_data = load_json_file('outputs/Output_results_capacities.json')
        energy_flows_data = load_json_file('outputs/Output_session.json')

        # Plot Levelized Cost of Steel
        if "levelized_cost_AUD_per_tonne" in lcos_data:
            plot_levelized_cost(lcos_data["levelized_cost_AUD_per_tonne"], save_path='levelized_cost_plot.png')
        else:
            print("Error: Levelized cost data not found in Output_results.json")

        # Plot Installed Capacities
        if capacities_data:
            plot_installed_capacities(capacities_data, save_path='installed_capacities_plot.png')
        else:
            print("Error: Installed capacities data not found in Output_results_capacities.json")

        # Plot Energy Flows
        if energy_flows_data:
            plot_energy_flows(energy_flows_data, save_path='energy_flows_plot.png')
        else:
            print("Error: Energy flows data not found in Output_session.json")

    except FileNotFoundError as e:
        print(f"Error: {e} - One or more output files are missing.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
