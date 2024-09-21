import numpy as np
import json
import os
from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GreenSteelModel:
    def __init__(self, config: dict):
        """
        Initialize the GreenSteelModel with configuration parameters.
        
        Parameters:
            config (dict): Configuration parameters loaded from JSON.
        """
        self.config = config
        self.production_target = config.get("product_target_tonne_per_yr", 0)
        self.product = config.get("product", "steel")
        
        # Initialize system capacities (MW)
        self.wind_capacity = config.get("wind_capacity_MW", 0)
        self.solar_capacity = config.get("solar_capacity_MW", 0)
        self.electro_capacity = config.get("electro_MW", 0)
        self.battery_capacity = config.get("battery_MW", 0)
        
        # Battery storage parameters
        self.battery_storage_hours = config.get("BESS_hours_of_storage", 0)
        self.battery_energy_capacity = self.battery_capacity * self.battery_storage_hours  # MWh
        
        # Initialize cost assumptions
        self.cost_assumptions = {
            "wind_capex_kW": config.get("Wind_capex_kW", 0),
            "solar_capex_kW": config.get("PV_capex_kW", 0),
            "battery_capex_kW": config.get("BESS_capex_kW", 0),
            "electrolyser_capex_kW": config.get("electrolyser_capex_kW", 0),
            "h2_storage_capex_tonne": config.get("h2_storage_capex_tonne", 0),
            "capex_h2Shaft_ton_euro": config.get("capex_h2Shaft_ton_euro", 0),
            "capex_EAF_ton_euro": config.get("capex_EAF_ton_euro", 0)
        }
        
        # Initialize results
        self.results = {
            "levelized_cost": 0,
            "installed_capacities": {}
        }

    @classmethod
    def from_json_file(cls, filepath: str) -> 'GreenSteelModel':
        """Create an instance of GreenSteelModel from a JSON configuration file."""
        with open(filepath, 'r') as file:
            config = json.load(file)
        return cls(config)

    def initialize(self):
        """Initialize the model by loading data and setting up initial parameters."""
        logging.info("Initializing the model...")
        self.wind_data = self.load_energy_data(self.config.get("wind_input_CF_filename", "wind.csv"))
        self.solar_data = self.load_energy_data(self.config.get("solar_input_CF_filename", "pv.csv"))
        
        # Initialize installed capacities
        self.results["installed_capacities"] = {
            "Wind power (MW)": self.wind_capacity,
            "Solar power (MW)": self.solar_capacity,
            "Battery storage (MW)": self.battery_capacity,
            "Electrolyser (MW)": self.electro_capacity,
            "Hydrogen tank (tonne)": 0  # To be calculated later
        }

    def load_energy_data(self, filepath: str) -> np.ndarray:
        """Load energy capacity factor data from a CSV file."""
        if not os.path.exists(filepath):
            logging.error(f"Energy data file not found: {filepath}")
            raise FileNotFoundError(f"Energy data file not found: {filepath}")
        return np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=1)

    def simulate_energy_flows(self, hours: int = 8760) -> dict:
        """Simulate hourly energy flows for the entire year."""
        wind_cf = np.resize(self.wind_data, hours)
        solar_cf = np.resize(self.solar_data, hours)

        wind_generation = self.wind_capacity * wind_cf
        solar_generation = self.solar_capacity * solar_cf
        total_generation = wind_generation + solar_generation

        battery_charge = np.zeros(hours)
        battery_discharge = np.zeros(hours)
        battery_level = np.zeros(hours)
        hydrogen_level = np.zeros(hours)

        battery_efficiency = 0.9
        hydrogen_production_efficiency = 0.7
        depth_of_discharge = 0.8

        for h in range(1, hours):
            excess_energy = total_generation[h] - self.electro_capacity
            if excess_energy > 0:
                charge = min(excess_energy * battery_efficiency, self.battery_energy_capacity - battery_level[h-1])
                battery_charge[h] = charge
                battery_level[h] += charge
            else:
                discharge = min(-excess_energy, battery_level[h-1] * depth_of_discharge)
                battery_discharge[h] = discharge
                battery_level[h] -= discharge

        hydrogen_production = self.electro_capacity * hydrogen_production_efficiency
        hydrogen_level = np.clip(np.cumsum(hydrogen_production), 0, self.battery_energy_capacity)

        energy_flows = {
            "Wind generation (MWh)": wind_generation,
            "Solar generation (MWh)": solar_generation,
            "Battery charge (MWh)": battery_charge,
            "Battery discharge (MWh)": battery_discharge,
            "Hydrogen production (tonne)": hydrogen_production
        }

        self.results["energy_flows"] = energy_flows
        return energy_flows

    def calculate_costs(self):
        """Calculate the levelized cost of steel (LCOS) and CAPEX."""
        wind_capex = self.wind_capacity * self.cost_assumptions["wind_capex_kW"] / 1000
        solar_capex = self.solar_capacity * self.cost_assumptions["solar_capex_kW"] / 1000
        battery_capex = self.battery_capacity * self.cost_assumptions["battery_capex_kW"] / 1000
        electro_capex = self.electro_capacity * self.cost_assumptions["electrolyser_capex_kW"] / 1000
        h2_storage_capex = self.results["installed_capacities"]["Hydrogen tank (tonne)"] * self.cost_assumptions["h2_storage_capex_tonne"]

        capex_dri_eaf = self.cost_assumptions["capex_h2Shaft_ton_euro"] * 1.6 * self.production_target
        total_capex = wind_capex + solar_capex + battery_capex + electro_capex + h2_storage_capex + capex_dri_eaf

        lcos = total_capex / self.production_target
        self.results["levelized_cost"] = lcos
        logging.info(f"Levelized Cost of Steel (LCOS): {lcos:.2f} AUD/tonne")

    def save_results(self):
        """Save simulation results to JSON files."""
        with open('Output_session.json', 'w') as file:
            json.dump(self.results["energy_flows"], file, indent=4)

        with open('Output_results.json', 'w') as file:
            json.dump({"levelized_cost_AUD_per_tonne": self.results["levelized_cost"]}, file, indent=4)

        with open('Output_results_capacities.json', 'w') as file:
            json.dump(self.results["installed_capacities"], file, indent=4)

    def optimize_system(self):
        """Optimize system capacities to minimize the LCOS."""
        def objective_function(capacities):
            self.wind_capacity, self.solar_capacity, self.battery_capacity = capacities
            self.simulate_energy_flows()
            self.calculate_costs()
            return self.results["levelized_cost"]

        initial_guess = [self.wind_capacity, self.solar_capacity, self.battery_capacity]
        bounds = [(100, 1000), (100, 1000), (100, 500)]  # Wind, solar, battery bounds

        result = minimize(objective_function, initial_guess, bounds=bounds)
        optimal_capacities = result.x
        self.wind_capacity, self.solar_capacity, self.battery_capacity = optimal_capacities
        logging.info(f"Optimized capacities: {optimal_capacities}")

if __name__ == "__main__":
    model = GreenSteelModel.from_json_file("config.json")
    model.initialize()
    model.simulate_energy_flows()
    model.calculate_costs()
    model.save_results()
    model.optimize_system()
