import numpy as np
import json
import os

class GreenSteelModel:
    def __init__(self, config):
        """
        Initialize the GreenSteelModel with configuration parameters.
        
        Parameters:
            config (dict): Configuration parameters loaded from JSON.
        """
        self.config = config
        self.production_target = config.get("product_target_tonne_per_yr", 0)  # in tonnes per year
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
        
        # Load renewable energy data
        self.wind_data = None
        self.solar_data = None
        
        # Initialize results
        self.results = {
            "levelized_cost": 0,
            "installed_capacities": {}
        }

    @classmethod
    def FromJsonFile(cls, filepath):
        """
        Create an instance of GreenSteelModel from a JSON configuration file.
        
        Parameters:
            filepath (str): Path to the JSON configuration file.
        
        Returns:
            GreenSteelModel: An instance of the model.
        """
        with open(filepath, 'r') as file:
            config = json.load(file)
        return cls(config)

    def Initialize(self):
        """
        Initialize the model by loading data and setting up initial parameters.
        """
        # Load renewable energy data
        self.wind_data = self.load_energy_data(self.config.get("wind_input_CF_filename", "wind.csv"))
        self.solar_data = self.load_energy_data(self.config.get("solar_input_CF_filename", "pv.csv"))
        
        # Initialize installed capacities
        self.results["installed_capacities"] = {
            "Wind power (MW)": self.wind_capacity,
            "Solar power (MW)": self.solar_capacity,
            "Battery storage (MW)": self.battery_capacity,
            "Electrolyser (MW)": self.electro_capacity,
            "Hydrogen tank (tonne)": 0  # To be calculated
        }

    def load_energy_data(self, filepath):
        """
        Load energy capacity factor data from a CSV file.
        
        Parameters:
            filepath (str): Path to the CSV file.
        
        Returns:
            np.ndarray: Array of capacity factors.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Energy data file not found: {filepath}")
        data = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=1)
        return data

    def Run(self):
        """
        Run the simulation of the green steel production pathway.
        """
        # Simulate hourly energy flows
        hours = 8760  # Number of hours in a year
        self.results["energy_flows"] = self.simulate_energy_flows(hours)
        
        # Calculate costs
        self.calculate_costs()
        
        # Save results
        self.save_results()

    def simulate_energy_flows(self, hours):
        """
        Simulate hourly energy flows for the entire year.
        
        Parameters:
            hours (int): Number of hours to simulate.
        
        Returns:
            dict: Dictionary containing energy flow arrays.
        """
        # Ensure data arrays are the correct length
        wind_cf = np.resize(self.wind_data, hours)  # Capacity factors for wind
        solar_cf = np.resize(self.solar_data, hours)  # Capacity factors for solar
        
        # Calculate generation in MWh
        wind_generation = self.wind_capacity * wind_cf  # MW * capacity factor = MW (assuming 1 hour)
        solar_generation = self.solar_capacity * solar_cf  # MW * capacity factor = MW
        
        # Total generation
        total_generation = wind_generation + solar_generation  # MW
        
        # Battery storage logic (simple charging/discharging)
        battery_charge = np.zeros(hours)
        battery_discharge = np.zeros(hours)
        battery_level = np.zeros(hours)
        
        for h in range(1, hours):
            excess_energy = total_generation[h] - self.electro_capacity
            if excess_energy > 0:
                # Charge the battery with excess energy, limited by storage capacity
                charge = min(excess_energy, self.battery_energy_capacity - battery_level[h-1])
                battery_charge[h] = charge
                battery_level[h] = battery_level[h-1] + charge
            else:
                # Discharge the battery to meet the deficit
                discharge = min(-excess_energy, battery_level[h-1])
                battery_discharge[h] = discharge
                battery_level[h] = battery_level[h-1] - discharge
        
        # Hydrogen production
        hydrogen_production = self.electro_capacity * 24  # Assuming electrolyzers operate continuously
        
        # Hydrogen usage in steel production
        steel_production = hydrogen_production * self.config.get("Iron_ore_to_steel_conversion", 1.5)  # Tonnes of steel
        
        # Store energy flows
        energy_flows = {
            "Wind generation (MWh)": wind_generation,
            "Solar generation (MWh)": solar_generation,
            "Battery charge (MWh)": battery_charge,
            "Battery discharge (MWh)": battery_discharge,
            "Electrolysis (MWh)": self.electro_capacity,
            "Hydrogen production (tonne)": hydrogen_production,
            "Steel production (tonne)": steel_production
        }
        
        return energy_flows

    def calculate_costs(self):
        """
        Calculate the levelized cost of steel and total CAPEX.
        """
        # Calculate CAPEX for each component
        wind_capex = self.wind_capacity * self.cost_assumptions["wind_capex_kW"] / 1000  # MW * kW/MW
        solar_capex = self.solar_capacity * self.cost_assumptions["solar_capex_kW"] / 1000  # MW * kW/MW
        battery_capex = self.battery_capacity * self.cost_assumptions["battery_capex_kW"] / 1000  # MW * kW/MW
        electro_capex = self.electro_capacity * self.cost_assumptions["electrolyser_capex_kW"] / 1000  # MW * kW/MW
        h2_storage_capex = self.results["installed_capacities"]["Hydrogen tank (tonne)"] * self.cost_assumptions["h2_storage_capex_tonne"]  # tonne * €/tonne
        
        # CAPEX for DRI and EAF
        capex_dri_eaf = self.cost_assumptions["capex_h2Shaft_ton_euro"] * 1.6 * self.production_target  # € to AUD
        
        # Total CAPEX
        total_capex = wind_capex + solar_capex + battery_capex + electro_capex + h2_storage_capex + capex_dri_eaf
        
        # Calculate Levelized Cost of Steel (LCOS)
        lcos = total_capex / self.production_target  # AUD per tonne
        
        self.results["levelized_cost"] = lcos

    def save_results(self):
        """
        Save simulation results to JSON files.
        """
        # Save energy flows
        with open('Output_session.json', 'w') as file:
            json.dump(self.results["energy_flows"], file, indent=4)
        
        # Save results summary
        with open('Output_results.json', 'w') as file:
            json.dump({"levelized_cost_AUD_per_tonne": self.results["levelized_cost"]}, file, indent=4)
        
        # Save installed capacities
        with open('Output_results_capacities.json', 'w') as file:
            json.dump(self.results["installed_capacities"], file, indent=4)

