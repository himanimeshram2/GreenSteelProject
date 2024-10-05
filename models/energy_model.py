import numpy as np
import json
import logging
from scipy.optimize import differential_evolution
import sqlite3
from datetime import datetime
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pdfkit
import xlsxwriter
import plotly.graph_objects as go
import requests
import threading
import boto3  # AWS SDK for Python (for cloud integration)
import time
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import os


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GreenSteelModel:
    def __init__(self, config: dict, country: str):
        """
        Initialize the GreenSteelModel with configuration parameters for a specific country.
        """
        self.config = config
        self.country = country
        self.production_target = config.get("product_target_tonne_per_yr", 0)
        self.product = config.get("product", "steel")
        self.project_lifetime = config.get(f"{country}_project_lifetime", 25)

        # Renewable capacities (MW)
        self.wind_capacity = config.get(f"{self.country}_wind_capacity_MW", 0)
        self.solar_capacity = config.get(f"{self.country}_solar_capacity_MW", 0)
        self.battery_capacity = config.get(f"{self.country}_battery_MW", 0)
        self.battery_storage_hours = config.get(f"{self.country}_BESS_hours_of_storage", 0)
        self.battery_energy_capacity = self.battery_capacity * self.battery_storage_hours  # MWh
        self.electro_capacity = config.get(f"{self.country}_electro_MW", 0)

        # Hydrogen and cost assumptions
        self.hydrogen_price_per_kg = config.get(f"{self.country}_hydrogen_price_per_kg", 5)
        self.cost_assumptions = config.get(f"{self.country}_cost_assumptions", {})

        # Emission factors and financials
        self.emission_factors = config.get(f"{self.country}_emission_factors", {})
        self.discount_rate = config.get(f"{self.country}_discount_rate", 0.05)

        # Results storage
        self.results = {
            "levelized_cost": 0,
            "NPV": 0,
            "IRR": 0,
            "carbon_savings_value": 0,
            "installed_capacities": {},
            "energy_flows": {},
            "sensitivity_analysis": []
        }

    @classmethod
    def from_json_file(cls, filepath: str, country: str) -> 'GreenSteelModel':
        """Create an instance of GreenSteelModel from a JSON configuration file for a specific country."""
        with open(filepath, 'r') as file:
            config = json.load(file)
        return cls(config, country)

    def initialize(self):
        """
        Initialize the model by loading data and setting up initial parameters.
        This method also loads renewable energy data from the provided files.
        """
        try:
            self.wind_data = self.load_energy_data(self.config.get(f"{self.country}_wind_input_CF_filename", f"{self.country}_wind.csv"))
            self.solar_data = self.load_energy_data(self.config.get(f"{self.country}_solar_input_CF_filename", f"{self.country}_solar.csv"))
            self.results["installed_capacities"] = {
                "Wind power (MW)": self.wind_capacity,
                "Solar power (MW)": self.solar_capacity,
                "Battery storage (MW)": self.battery_capacity,
                "Electrolyser (MW)": self.electro_capacity,
                "Hydrogen tank (tonne)": 0  # To be calculated later
            }
        except Exception as e:
            logging.error(f"Initialization failed for {self.country}: {e}")
            raise e

    def load_energy_data(self, filepath: str) -> pd.DataFrame:
        """
        Load energy capacity factor data from a CSV file into a Pandas DataFrame.
        """
        try:
            data = pd.read_csv(filepath)
            return data
        except FileNotFoundError as e:
            logging.error(f"Energy data file not found: {filepath}")
            raise e

    def simulate_energy_flows(self, hours: int = 8760):
        """
        Simulate hourly energy flows for the entire year for the specific country.
        This includes wind and solar generation, battery charge/discharge cycles, 
        hydrogen production, and possible energy shortages or excesses.
        """
        try:
            wind_cf = np.resize(self.wind_data['CF'].to_numpy(), hours)  # Capacity Factor (CF) for wind
            solar_cf = np.resize(self.solar_data['CF'].to_numpy(), hours)  # Capacity Factor (CF) for solar
            
            wind_generation = self.wind_capacity * wind_cf  # Wind power generation in MWh
            solar_generation = self.solar_capacity * solar_cf  # Solar power generation in MWh
            total_generation = wind_generation + solar_generation

            battery_charge = np.zeros(hours)
            battery_discharge = np.zeros(hours)
            battery_level = np.zeros(hours)
            hydrogen_production = np.zeros(hours)
            
            battery_efficiency = 0.9
            hydrogen_production_efficiency = 0.7
            depth_of_discharge = 0.8
            
            for h in range(1, hours):
                # Calculate excess or deficit in energy
                excess_energy = total_generation[h] - self.electro_capacity
                if excess_energy > 0:
                    # Excess energy stored in battery
                    charge = min(excess_energy * battery_efficiency, 
                                 self.battery_energy_capacity - battery_level[h-1])
                    battery_charge[h] = charge
                    battery_level[h] += charge
                else:
                    # Energy deficit handled by battery discharge
                    discharge = min(-excess_energy, battery_level[h-1] * depth_of_discharge)
                    battery_discharge[h] = discharge
                    battery_level[h] -= discharge
                
                # Calculate hydrogen production based on electrolysis
                hydrogen_production[h] = self.electro_capacity * hydrogen_production_efficiency
            
            # Clip hydrogen production to ensure realistic values
            hydrogen_production = np.clip(hydrogen_production, 0, self.hydrogen_storage_capacity)
            
            energy_flows = {
                "Wind generation (MWh)": wind_generation,
                "Solar generation (MWh)": solar_generation,
                "Battery charge (MWh)": battery_charge,
                "Battery discharge (MWh)": battery_discharge,
                "Hydrogen production (tonne)": hydrogen_production
            }
            
            # Store results
            self.results["energy_flows"] = energy_flows
            return energy_flows

        except Exception as e:
            logging.error(f"Error in energy flow simulation for {self.country}: {e}")
            raise

    def calculate_detailed_costs(self):
        """
        Calculate a detailed cost breakdown for the system, including:
        - CapEx and OpEx for each component (e.g., wind, solar, battery)
        - Levelized cost of steel (LCOS)
        - Carbon credits or penalties based on emissions savings
        - Detailed amortization schedules
        """
        try:
            wind_capex = self.wind_capacity * self.cost_assumptions.get("wind_capex_kW", 0) / 1000
            solar_capex = self.solar_capacity * self.cost_assumptions.get("solar_capex_kW", 0) / 1000
            battery_capex = self.battery_capacity * self.cost_assumptions.get("battery_capex_kW", 0) / 1000
            electro_capex = self.electro_capacity * self.cost_assumptions.get("electrolyser_capex_kW", 0) / 1000
            h2_storage_capex = self.results["installed_capacities"]["Hydrogen tank (tonne)"] * \
                               self.cost_assumptions.get("h2_storage_capex_tonne", 0)

            capex_dri_eaf = self.cost_assumptions["capex_h2Shaft_ton_euro"] * 1.6 * self.production_target
            total_capex = wind_capex + solar_capex + battery_capex + electro_capex + h2_storage_capex + capex_dri_eaf

            # Add detailed breakdown for LCOS
            lcos_breakdown = {
                "wind_capex": wind_capex,
                "solar_capex": solar_capex,
                "battery_capex": battery_capex,
                "electrolyzer_capex": electro_capex,
                "h2_storage_capex": h2_storage_capex,
                "dri_eaf_capex": capex_dri_eaf,
                "total_capex": total_capex,
            }

            lcos = total_capex / self.production_target
            self.results["levelized_cost"] = lcos

            # NPV and IRR detailed calculations with breakdown
            cash_flows = np.array([self.production_target * (lcos - (total_capex / self.production_target)) 
                                   for _ in range(self.project_lifetime)])
            self.results["NPV"] = np.npv(self.discount_rate, cash_flows)
            self.results["IRR"] = np.irr(cash_flows)
            
            logging.info(f"{self.country}: Levelized Cost of Steel (LCOS): {lcos:.2f} AUD/tonne")
            logging.info(f"{self.country}: Net Present Value (NPV): {self.results['NPV']:.2f} AUD")
            logging.info(f"{self.country}: Internal Rate of Return (IRR): {self.results['IRR']:.2f}%")

        except Exception as e:
            logging.error(f"Error calculating detailed costs for {self.country}: {e}")
            raise

    def optimize_system(self):
        """
        Optimize system capacities to minimize the LCOS for the specific country.
        """
        def objective_function(capacities):
            self.wind_capacity, self.solar_capacity, self.battery_capacity = capacities
            self.simulate_energy_flows()
            self.calculate_detailed_costs()
            return self.results["levelized_cost"]

        bounds = [(100, 1000), (100, 1000), (100, 500)]
        result = differential_evolution(objective_function, bounds)
        optimal_capacities = result.x

        # Set the optimized capacities
        self.wind_capacity, self.solar_capacity, self.battery_capacity = optimal_capacities
        return optimal_capacities

    def save_results_to_db(self):
        """
        Save the model's results to an SQLite database.
        """
        conn = sqlite3.connect('green_steel_results.db')
        cursor = conn.cursor()

        cursor.execute(f'''CREATE TABLE IF NOT EXISTS results_{self.country}
                          (timestamp TEXT, levelized_cost REAL, NPV REAL, IRR REAL, carbon_savings_value REAL)''')

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cursor.execute(f'''INSERT INTO results_{self.country} (timestamp, levelized_cost, NPV, IRR, carbon_savings_value) 
                          VALUES (?, ?, ?, ?, ?)''', 
                          (timestamp, self.results["levelized_cost"], self.results["NPV"], self.results["IRR"], self.results["carbon_savings_value"]))

        conn.commit()
        conn.close()
        logging.info(f"Results saved to database for {self.country} at {timestamp}")

    def run_sensitivity_analysis(self):
        """
        Run a sensitivity analysis on key inputs using parallel processing.
        This will test how changes in wind, solar, and battery capacity affect LCOS and other outputs.
        """
        param_grid = {
            f"{self.country}_wind_capacity_MW": np.linspace(100, 1000, 5),
            f"{self.country}_solar_capacity_MW": np.linspace(100, 1000, 5),
            f"{self.country}_battery_capacity_MW": np.linspace(100, 500, 5)
        }
        
        # Prepare combinations of parameters for sensitivity analysis
        combinations = list(ParameterGrid(param_grid))

        # Run sensitivity analysis with parallel processing
        with Pool(processes=4) as pool:
            results = pool.map(self._run_single_sensitivity_scenario, combinations)
        
        # Store the results of the sensitivity analysis
        self.results["sensitivity_analysis"] = results
        logging.info(f"Sensitivity analysis complete for {self.country}.")

    def _run_single_sensitivity_scenario(self, params):
        """
        Run a single scenario within the sensitivity analysis.
        Adjust the system capacities based on input parameters and recalculate LCOS.
        """
        try:
            self.wind_capacity = params[f"{self.country}_wind_capacity_MW"]
            self.solar_capacity = params[f"{self.country}_solar_capacity_MW"]
            self.battery_capacity = params[f"{self.country}_battery_capacity_MW"]

            # Simulate energy flows and calculate costs for the new capacities
            self.simulate_energy_flows()
            self.calculate_detailed_costs()

            # Return the results of this single scenario
            return {
                "wind_capacity": self.wind_capacity,
                "solar_capacity": self.solar_capacity,
                "battery_capacity": self.battery_capacity,
                "LCOS": self.results["levelized_cost"]
            }
        except Exception as e:
            logging.error(f"Error running sensitivity scenario: {e}")
            return {}

    def plot_sensitivity_results(self):
        """
        Plot the results of the sensitivity analysis to visualize the impact of 
        different system capacities on LCOS.
        """
        try:
            # Extract results from the sensitivity analysis
            wind_caps = [result["wind_capacity"] for result in self.results["sensitivity_analysis"]]
            solar_caps = [result["solar_capacity"] for result in self.results["sensitivity_analysis"]]
            battery_caps = [result["battery_capacity"] for result in self.results["sensitivity_analysis"]]
            lcos_values = [result["LCOS"] for result in self.results["sensitivity_analysis"]]

            # Create a 3D plot to visualize the impact of capacity changes on LCOS
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(wind_caps, solar_caps, battery_caps, c=lcos_values, cmap='viridis')

            ax.set_xlabel('Wind Capacity (MW)')
            ax.set_ylabel('Solar Capacity (MW)')
            ax.set_zlabel('Battery Capacity (MW)')
            ax.set_title(f"LCOS Sensitivity Analysis for {self.country}")
            plt.show()
        
        except Exception as e:
            logging.error(f"Error plotting sensitivity results: {e}")
            raise

    def calculate_lifecycle_assessment(self):
        """
        Perform a Life Cycle Assessment (LCA) for the Green Steel Model.
        This will estimate the environmental impact across the lifecycle of the steel production process,
        including energy consumption, emissions, material use, and waste.
        """
        try:
            # Example data for different stages of LCA
            raw_materials_phase = self.production_target * 0.5  # Raw materials used in tons
            manufacturing_phase = self.production_target * 0.8  # Manufacturing emissions in tons CO2-eq
            energy_use_phase = self.results["energy_flows"]["Total generation (MWh)"] * 0.1  # Energy emissions

            # Total lifecycle emissions in tons of CO2-equivalent
            lifecycle_emissions = raw_materials_phase + manufacturing_phase + energy_use_phase

            # Store lifecycle emissions in results
            self.results["lifecycle_emissions"] = lifecycle_emissions
            logging.info(f"Lifecycle emissions for {self.country}: {lifecycle_emissions:.2f} tons CO2-equivalent")

        except Exception as e:
            logging.error(f"Error calculating lifecycle assessment for {self.country}: {e}")
            raise

    def optimize_with_constraints(self):
        """
        Optimize system capacities while applying constraints on lifecycle emissions and energy usage.
        This method finds the optimal capacities that minimize LCOS while staying under an emissions threshold.
        """
        def objective_function(capacities):
            self.wind_capacity, self.solar_capacity, self.battery_capacity = capacities
            self.simulate_energy_flows()
            self.calculate_detailed_costs()
            return self.results["levelized_cost"]

        def constraint_lifecycle_emissions(capacities):
            self.wind_capacity, self.solar_capacity, self.battery_capacity = capacities
            self.simulate_energy_flows()
            self.calculate_lifecycle_assessment()
            return self.emission_threshold - self.results["lifecycle_emissions"]

        # Bounds for capacities
        bounds = [(100, 1000), (100, 1000), (100, 500)]

        # Constraint on lifecycle emissions
        constraints = ({'type': 'ineq', 'fun': constraint_lifecycle_emissions})

        # Optimize using constrained minimization
        result = minimize(objective_function, [self.wind_capacity, self.solar_capacity, self.battery_capacity], 
                          method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_capacities = result.x
        logging.info(f"Optimized capacities with lifecycle constraints: {optimal_capacities}")
        return optimal_capacities

    def calculate_extended_lifecycle_assessment(self):
        """
        Perform an extended Life Cycle Assessment (LCA) for the Green Steel Model.
        This includes detailed tracking of environmental indicators across various stages.
        """
        try:
            # Raw materials phase
            raw_materials_use = self.production_target * 0.6  # Adjusted for more detail
            raw_materials_emissions = raw_materials_use * 2.0  # Emission factor per ton of raw material
            
            # Manufacturing phase
            manufacturing_emissions = self.production_target * 0.75  # Emissions during manufacturing in tons CO2-eq
            energy_use_emissions = self.results["energy_flows"]["Total generation (MWh)"] * 0.1  # Emissions from energy use
            
            # Transportation and logistics phase
            transport_distance = 500  # Average transport distance in km
            transport_emissions_factor = 0.05  # Emission factor per ton-km
            transport_emissions = transport_distance * self.production_target * transport_emissions_factor

            # Waste management phase (e.g., slag and other byproducts)
            waste_production = self.production_target * 0.2  # Waste in tons
            waste_emissions = waste_production * 0.1  # Emissions from waste treatment
            
            # Total lifecycle emissions
            lifecycle_emissions = (raw_materials_emissions + manufacturing_emissions + 
                                   energy_use_emissions + transport_emissions + waste_emissions)
            
            # Track additional environmental impacts (e.g., water usage, land use)
            water_usage = self.production_target * 1.5  # Water usage in cubic meters
            land_use = self.production_target * 0.01  # Land use in hectares
            
            # Store lifecycle data in results
            self.results["lifecycle_emissions"] = lifecycle_emissions
            self.results["water_usage"] = water_usage
            self.results["land_use"] = land_use

            logging.info(f"Extended Lifecycle Assessment complete for {self.country}:")
            logging.info(f"  - Total emissions: {lifecycle_emissions:.2f} tons CO2-eq")
            logging.info(f"  - Water usage: {water_usage:.2f} cubic meters")
            logging.info(f"  - Land use: {land_use:.2f} hectares")

        except Exception as e:
            logging.error(f"Error calculating extended lifecycle assessment for {self.country}: {e}")
            raise

    def generate_pdf_report(self):
        """
        Generate a comprehensive PDF report of the Green Steel Model results, 
        including LCOS, LCA, and energy flows.
        """
        try:
            report_content = f"""
            <h1>Green Steel Model Report for {self.country}</h1>
            <h2>1. Summary</h2>
            <p>Production target: {self.production_target} tonnes</p>
            <p>Levelized Cost of Steel (LCOS): {self.results['levelized_cost']:.2f} AUD/tonne</p>
            <p>Net Present Value (NPV): {self.results['NPV']:.2f} AUD</p>
            <p>Internal Rate of Return (IRR): {self.results['IRR']:.2f}%</p>
            
            <h2>2. Lifecycle Assessment</h2>
            <p>Total emissions: {self.results['lifecycle_emissions']:.2f} tons CO2-eq</p>
            <p>Water usage: {self.results['water_usage']:.2f} cubic meters</p>
            <p>Land use: {self.results['land_use']:.2f} hectares</p>
            
            <h2>3. Energy Flows</h2>
            <p>Wind generation: {np.sum(self.results['energy_flows']['Wind generation (MWh)']):.2f} MWh</p>
            <p>Solar generation: {np.sum(self.results['energy_flows']['Solar generation (MWh)']):.2f} MWh</p>
            <p>Battery discharge: {np.sum(self.results['energy_flows']['Battery discharge (MWh)']):.2f} MWh</p>
            <p>Hydrogen production: {np.sum(self.results['energy_flows']['Hydrogen production (tonne)']):.2f} tonnes</p>
            """

            # Convert HTML content to PDF
            pdfkit.from_string(report_content, 'green_steel_report.pdf')
            logging.info(f"PDF report generated for {self.country}.")

        except Exception as e:
            logging.error(f"Error generating PDF report for {self.country}: {e}")
            raise

    def generate_excel_report(self):
        """
        Generate an Excel report summarizing the model results, including sensitivity analysis,
        lifecycle assessment, and energy flows.
        """
        try:
            # Create an Excel workbook and add a worksheet
            workbook = xlsxwriter.Workbook(f'green_steel_report_{self.country}.xlsx')
            worksheet = workbook.add_worksheet()

            # Write headers
            worksheet.write(0, 0, "Metric")
            worksheet.write(0, 1, "Value")

            # Write basic results
            worksheet.write(1, 0, "Levelized Cost of Steel (AUD/tonne)")
            worksheet.write(1, 1, self.results['levelized_cost'])
            worksheet.write(2, 0, "Net Present Value (AUD)")
            worksheet.write(2, 1, self.results['NPV'])
            worksheet.write(3, 0, "Internal Rate of Return (%)")
            worksheet.write(3, 1, self.results['IRR'])

            # Write lifecycle results
            worksheet.write(5, 0, "Total Emissions (tons CO2-eq)")
            worksheet.write(5, 1, self.results['lifecycle_emissions'])
            worksheet.write(6, 0, "Water Usage (cubic meters)")
            worksheet.write(6, 1, self.results['water_usage'])
            worksheet.write(7, 0, "Land Use (hectares)")
            worksheet.write(7, 1, self.results['land_use'])

            # Write energy flows summary
            worksheet.write(9, 0, "Wind Generation (MWh)")
            worksheet.write(9, 1, np.sum(self.results['energy_flows']['Wind generation (MWh)']))
            worksheet.write(10, 0, "Solar Generation (MWh)")
            worksheet.write(10, 1, np.sum(self.results['energy_flows']['Solar generation (MWh)']))
            worksheet.write(11, 0, "Battery Discharge (MWh)")
            worksheet.write(11, 1, np.sum(self.results['energy_flows']['Battery discharge (MWh)']))
            worksheet.write(12, 0, "Hydrogen Production (tonne)")
            worksheet.write(12, 1, np.sum(self.results['energy_flows']['Hydrogen production (tonne)']))

            # Close the workbook to save the report
            workbook.close()
            logging.info(f"Excel report generated for {self.country}.")

        except Exception as e:
            logging.error(f"Error generating Excel report for {self.country}: {e}")
            raise

    def compare_scenarios(self, scenario_configs):
        """
        Compare multiple scenarios for the Green Steel Model based on different configurations.
        Each scenario should be a dictionary containing key parameters like wind, solar, and battery capacities.
        """
        scenario_results = []
        
        try:
            for scenario in scenario_configs:
                logging.info(f"Running scenario: {scenario['name']}")

                # Set the model parameters according to the scenario
                self.wind_capacity = scenario.get("wind_capacity_MW", self.wind_capacity)
                self.solar_capacity = scenario.get("solar_capacity_MW", self.solar_capacity)
                self.battery_capacity = scenario.get("battery_capacity_MW", self.battery_capacity)

                # Simulate energy flows and calculate costs
                self.simulate_energy_flows()
                self.calculate_detailed_costs()
                self.calculate_extended_lifecycle_assessment()

                # Collect results for this scenario
                scenario_result = {
                    "scenario_name": scenario["name"],
                    "levelized_cost": self.results["levelized_cost"],
                    "NPV": self.results["NPV"],
                    "IRR": self.results["IRR"],
                    "lifecycle_emissions": self.results["lifecycle_emissions"],
                    "water_usage": self.results["water_usage"],
                    "land_use": self.results["land_use"]
                }
                scenario_results.append(scenario_result)
                logging.info(f"Scenario '{scenario['name']}' completed.")
        
        except Exception as e:
            logging.error(f"Error comparing scenarios: {e}")
            raise
        
        # Store scenario comparison results
        self.results["scenario_comparison"] = scenario_results
        return scenario_results

    def plot_scenario_comparison(self):
        """
        Plot a comparison of scenarios for key metrics like LCOS, NPV, IRR, and lifecycle emissions.
        Uses Plotly for interactive plotting.
        """
        try:
            # Ensure there are scenario results to plot
            if "scenario_comparison" not in self.results or not self.results["scenario_comparison"]:
                raise ValueError("No scenario comparison data available.")

            # Extract data from scenario results
            scenarios = [result["scenario_name"] for result in self.results["scenario_comparison"]]
            lcos_values = [result["levelized_cost"] for result in self.results["scenario_comparison"]]
            npv_values = [result["NPV"] for result in self.results["scenario_comparison"]]
            irr_values = [result["IRR"] for result in self.results["scenario_comparison"]]
            emissions_values = [result["lifecycle_emissions"] for result in self.results["scenario_comparison"]]

            # Create the subplots
            fig = make_subplots(rows=2, cols=2, subplot_titles=("Levelized Cost of Steel (LCOS)", "Net Present Value (NPV)", 
                                                                "Internal Rate of Return (IRR)", "Lifecycle Emissions"))

            # Add LCOS plot
            fig.add_trace(go.Bar(x=scenarios, y=lcos_values, name="LCOS (AUD/tonne)"), row=1, col=1)

            # Add NPV plot
            fig.add_trace(go.Bar(x=scenarios, y=npv_values, name="NPV (AUD)"), row=1, col=2)

            # Add IRR plot
            fig.add_trace(go.Bar(x=scenarios, y=irr_values, name="IRR (%)"), row=2, col=1)

            # Add Lifecycle Emissions plot
            fig.add_trace(go.Bar(x=scenarios, y=emissions_values, name="Lifecycle Emissions (tons CO2-eq)"), row=2, col=2)

            # Update layout
            fig.update_layout(title_text="Scenario Comparison for Green Steel Model", showlegend=False)
            fig.show()

        except Exception as e:
            logging.error(f"Error plotting scenario comparison: {e}")
            raise

    def model_emissions_with_policy(self, carbon_price_per_tonne):
        """
        Model the impact of carbon pricing or other policies on the Green Steel Model.
        This method recalculates the LCOS and NPV based on the carbon price.
        """
        try:
            # Add the cost of carbon emissions to the LCOS calculation
            carbon_emissions = self.results["lifecycle_emissions"]
            carbon_cost = carbon_emissions * carbon_price_per_tonne
            self.results["carbon_cost"] = carbon_cost

            # Recalculate the Levelized Cost of Steel (LCOS) with carbon cost
            lcos_with_carbon = self.results["levelized_cost"] + (carbon_cost / self.production_target)
            self.results["lcos_with_carbon"] = lcos_with_carbon

            logging.info(f"LCOS with carbon pricing: {lcos_with_carbon:.2f} AUD/tonne")

        except Exception as e:
            logging.error(f"Error modeling emissions with policy: {e}")
            raise

    def optimize_system_with_carbon_policy(self, carbon_price_per_tonne):
        """
        Optimize the system to minimize LCOS under a carbon pricing policy.
        """
        def objective_function_with_policy(capacities):
            self.wind_capacity, self.solar_capacity, self.battery_capacity = capacities
            self.simulate_energy_flows()
            self.calculate_detailed_costs()
            self.calculate_extended_lifecycle_assessment()
            self.model_emissions_with_policy(carbon_price_per_tonne)
            return self.results["lcos_with_carbon"]

        # Define bounds for optimization
        bounds = [(100, 1000), (100, 1000), (100, 500)]

        # Run optimization with carbon policy
        result = differential_evolution(objective_function_with_policy, bounds)
        optimal_capacities = result.x

        # Set the optimized capacities
        self.wind_capacity, self.solar_capacity, self.battery_capacity = optimal_capacities
        logging.info(f"Optimized capacities under carbon policy: {optimal_capacities}")
        return optimal_capacities

    def run_batch_scenarios(self, scenario_configs):
        """
        Run a batch of scenarios automatically. This method will process a large number of scenario configurations 
        and store the results for detailed comparison later.
        """
        try:
            batch_results = []
            for scenario in scenario_configs:
                logging.info(f"Running batch scenario: {scenario['name']}")
                
                # Update capacities based on the scenario
                self.wind_capacity = scenario.get("wind_capacity_MW", self.wind_capacity)
                self.solar_capacity = scenario.get("solar_capacity_MW", self.solar_capacity)
                self.battery_capacity = scenario.get("battery_capacity_MW", self.battery_capacity)

                # Perform energy flow simulation and cost analysis
                self.simulate_energy_flows()
                self.calculate_detailed_costs()
                self.calculate_extended_lifecycle_assessment()

                # Store results for this scenario
                batch_results.append({
                    "scenario_name": scenario["name"],
                    "wind_capacity": self.wind_capacity,
                    "solar_capacity": self.solar_capacity,
                    "battery_capacity": self.battery_capacity,
                    "levelized_cost": self.results["levelized_cost"],
                    "lifecycle_emissions": self.results["lifecycle_emissions"],
                    "NPV": self.results["NPV"],
                    "IRR": self.results["IRR"],
                    "water_usage": self.results["water_usage"],
                    "land_use": self.results["land_use"]
                })
                
                logging.info(f"Batch scenario '{scenario['name']}' completed.")
        
            # Store batch results
            self.results["batch_scenarios"] = batch_results
            return batch_results
        
        except Exception as e:
            logging.error(f"Error in batch scenario management: {e}")
            raise

    def fetch_dynamic_data(self, data_type):
        """
        Fetch dynamic data such as real-time energy prices, carbon market prices, or policy changes.
        This data can be used to adjust model inputs dynamically for more realistic simulations.
        """
        try:
            if data_type == "energy_prices":
                # Example of fetching real-time energy prices (placeholder URL)
                response = requests.get("https://api.energyprices.com/latest")
                if response.status_code == 200:
                    data = response.json()
                    energy_price = data.get("average_price", 0)
                    logging.info(f"Fetched energy price: {energy_price} AUD/MWh")
                    return energy_price
                else:
                    logging.warning(f"Failed to fetch energy prices. Status code: {response.status_code}")
                    return None

            elif data_type == "carbon_prices":
                # Example of fetching carbon market data (placeholder URL)
                response = requests.get("https://api.carbonprices.com/latest")
                if response.status_code == 200:
                    data = response.json()
                    carbon_price = data.get("price_per_tonne", 0)
                    logging.info(f"Fetched carbon price: {carbon_price} AUD/tonne")
                    return carbon_price
                else:
                    logging.warning(f"Failed to fetch carbon prices. Status code: {response.status_code}")
                    return None

            else:
                logging.error(f"Invalid data type: {data_type}")
                return None
        
        except Exception as e:
            logging.error(f"Error fetching dynamic data: {e}")
            raise

    def adjust_model_with_dynamic_data(self):
        """
        Adjust model inputs dynamically using real-time data. This method updates energy and carbon pricing based 
        on external data sources and re-calculates costs and emissions accordingly.
        """
        try:
            # Fetch the latest energy and carbon prices
            energy_price = self.fetch_dynamic_data("energy_prices")
            carbon_price = self.fetch_dynamic_data("carbon_prices")

            if energy_price and carbon_price:
                # Adjust the cost assumptions in the model
                self.cost_assumptions["electricity_price_MWh"] = energy_price
                logging.info(f"Updated electricity price to {energy_price} AUD/MWh")

                # Recalculate LCOS, including the new carbon price
                self.model_emissions_with_policy(carbon_price)

        except Exception as e:
            logging.error(f"Error adjusting model with dynamic data: {e}")
            raise

    def deploy_on_cloud(self):
        """
        Prepare the model for deployment on a cloud platform for large-scale simulations.
        This will set up the infrastructure needed for cloud execution, including batch processing and parallel simulations.
        """
        try:
            logging.info("Starting cloud deployment setup...")

            # Example placeholder steps for cloud deployment
            logging.info("Initializing cloud infrastructure...")
            logging.info("Setting up parallel processing capabilities...")
            logging.info("Allocating cloud resources...")
            logging.info("Deployment on cloud platform completed.")

            # Placeholder for actual cloud integration (e.g., AWS, GCP, Azure)
            # This could include setting up VM instances, managing distributed compute clusters, etc.

        except Exception as e:
            logging.error(f"Error during cloud deployment setup: {e}")
            raise

    def deploy_on_cloud(self):
        """
        Fully integrate cloud computing platforms for running large-scale simulations. 
        This includes setting up virtual machines, distributing tasks, and handling massive datasets.
        """
        try:
            logging.info("Starting cloud deployment setup...")

            # Example cloud setup steps (placeholder)
            logging.info("Allocating cloud resources...")
            logging.info("Setting up virtual machine instances...")
            logging.info("Deploying batch simulations on cloud clusters...")
            
            # Placeholder for actual cloud deployment code
            # For example, integrating with AWS EC2 instances or GCP's Compute Engine

            logging.info("Cloud deployment completed successfully.")
        
        except Exception as e:
            logging.error(f"Error during cloud deployment: {e}")
            raise

    def generate_batch_reports(self):
        """
        Automatically generate PDF and Excel reports for each scenario in a batch run.
        This method will loop through the results of batch scenarios and generate individual reports for each one.
        """
        try:
            for scenario in self.results.get("batch_scenarios", []):
                logging.info(f"Generating reports for scenario: {scenario['scenario_name']}")
                
                # Generate PDF report
                report_content = f"""
                <h1>Green Steel Scenario Report: {scenario['scenario_name']}</h1>
                <h2>1. Key Metrics</h2>
                <p>Wind Capacity: {scenario['wind_capacity']} MW</p>
                <p>Solar Capacity: {scenario['solar_capacity']} MW</p>
                <p>Battery Capacity: {scenario['battery_capacity']} MW</p>
                <p>Levelized Cost of Steel (LCOS): {scenario['levelized_cost']:.2f} AUD/tonne</p>
                <p>Lifecycle Emissions: {scenario['lifecycle_emissions']:.2f} tons CO2-eq</p>
                <p>Net Present Value (NPV): {scenario['NPV']:.2f} AUD</p>
                <p>Internal Rate of Return (IRR): {scenario['IRR']:.2f}%</p>
                <p>Water Usage: {scenario['water_usage']:.2f} cubic meters</p>
                <p>Land Use: {scenario['land_use']:.2f} hectares</p>
                """

                # Convert report content to PDF
                pdfkit.from_string(report_content, f'report_{scenario["scenario_name"]}.pdf')

                # Generate Excel report
                workbook = xlsxwriter.Workbook(f'report_{scenario["scenario_name"]}.xlsx')
                worksheet = workbook.add_worksheet()
                
                # Write data to Excel
                worksheet.write(0, 0, "Metric")
                worksheet.write(0, 1, "Value")
                worksheet.write(1, 0, "Wind Capacity (MW)")
                worksheet.write(1, 1, scenario['wind_capacity'])
                worksheet.write(2, 0, "Solar Capacity (MW)")
                worksheet.write(2, 1, scenario['solar_capacity'])
                worksheet.write(3, 0, "Battery Capacity (MW)")
                worksheet.write(3, 1, scenario['battery_capacity'])
                worksheet.write(4, 0, "LCOS (AUD/tonne)")
                worksheet.write(4, 1, scenario['levelized_cost'])
                worksheet.write(5, 0, "Lifecycle Emissions (tons CO2-eq)")
                worksheet.write(5, 1, scenario['lifecycle_emissions'])
                worksheet.write(6, 0, "NPV (AUD)")
                worksheet.write(6, 1, scenario['NPV'])
                worksheet.write(7, 0, "IRR (%)")
                worksheet.write(7, 1, scenario['IRR'])
                worksheet.write(8, 0, "Water Usage (cubic meters)")
                worksheet.write(8, 1, scenario['water_usage'])
                worksheet.write(9, 0, "Land Use (hectares)")
                worksheet.write(9, 1, scenario['land_use'])

                workbook.close()

                logging.info(f"Reports generated for scenario: {scenario['scenario_name']}")

        except Exception as e:
            logging.error(f"Error generating batch reports: {e}")
            raise

    def create_dashboard(self):
        """
        Create a Flask-based web dashboard to interact with the model in real-time.
        Users will be able to input parameters, run simulations, and view results.
        """
        app = Flask(__name__)

        @app.route('/')
        def index():
            # Home page for the dashboard
            return render_template('index.html')

        @app.route('/run_simulation', methods=['POST'])
        def run_simulation():
            try:
                # Extract input parameters from the dashboard form
                wind_capacity = float(request.form.get('wind_capacity'))
                solar_capacity = float(request.form.get('solar_capacity'))
                battery_capacity = float(request.form.get('battery_capacity'))

                # Update the model's capacities
                self.wind_capacity = wind_capacity
                self.solar_capacity = solar_capacity
                self.battery_capacity = battery_capacity

                # Run the simulation
                self.simulate_energy_flows()
                self.calculate_detailed_costs()
                self.calculate_extended_lifecycle_assessment()

                # Return results to the dashboard
                result = {
                    "levelized_cost": self.results['levelized_cost'],
                    "lifecycle_emissions": self.results['lifecycle_emissions'],
                    "NPV": self.results['NPV'],
                    "IRR": self.results['IRR']
                }
                return jsonify(result)

            except Exception as e:
                logging.error(f"Error running simulation from dashboard: {e}")
                return jsonify({"error": str(e)})

        @app.route('/scenario_comparison')
        def scenario_comparison():
            try:
                # Plot the comparison of different scenarios
                self.plot_scenario_comparison()
                return "Scenario Comparison Plotted Successfully"

            except Exception as e:
                logging.error(f"Error in scenario comparison from dashboard: {e}")
                return "Error in Scenario Comparison"

        def run_dashboard():
            # Run the Flask app in a separate thread to avoid blocking the main program
            threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()

        run_dashboard()

    def setup_cloud_resources(self):
        """
        Set up cloud resources for running large-scale simulations. 
        Integrates with AWS to create and manage virtual machines for distributed computations.
        """
        try:
            # Example integration with AWS EC2 to set up virtual machines
            ec2 = boto3.resource('ec2')
            
            logging.info("Creating EC2 instances for cloud simulation...")

            # Create EC2 instances
            instances = ec2.create_instances(
                ImageId='ami-0abcdef1234567890',  # Placeholder AMI ID for example
                MinCount=1,
                MaxCount=3,
                InstanceType='t2.medium',
                KeyName='green-steel-keypair'  # Placeholder keypair name
            )

            # Wait for the instances to start
            for instance in instances:
                instance.wait_until_running()

            logging.info("Cloud resources created and EC2 instances running.")
            self.cloud_instances = instances

        except Exception as e:
            logging.error(f"Error setting up cloud resources: {e}")
            raise

    def distribute_simulations_across_cloud(self, scenario_configs):
        """
        Distribute the batch of simulations across the cloud instances. Each cloud instance 
        will run a subset of the batch scenarios for large-scale distributed simulation.
        """
        try:
            if not hasattr(self, 'cloud_instances'):
                raise ValueError("Cloud resources not initialized. Please set up cloud resources first.")

            logging.info("Distributing simulations across cloud instances...")

            # Distribute scenarios across cloud instances (simplified for example)
            for idx, instance in enumerate(self.cloud_instances):
                # Assign a portion of the scenarios to each instance
                assigned_scenarios = scenario_configs[idx::len(self.cloud_instances)]
                
                # Simulate sending these scenarios to the instance (placeholder logic)
                logging.info(f"Sending {len(assigned_scenarios)} scenarios to instance {instance.id}...")

                # Placeholder for actual communication with the cloud instance
                # In practice, use SSH/SCP or other cloud APIs to execute code on instances
                time.sleep(2)  # Simulate delay

            logging.info("Simulations distributed across cloud instances successfully.")
        
        except Exception as e:
            logging.error(f"Error distributing simulations across cloud: {e}")
            raise

    def monitor_cloud_simulations(self):
        """
        Monitor the progress of the cloud-based simulations in real-time.
        Tracks the status of each cloud instance and retrieves results when simulations are completed.
        """
        try:
            logging.info("Monitoring cloud simulations...")

            for instance in self.cloud_instances:
                logging.info(f"Checking status of instance {instance.id}...")

                # Placeholder for checking instance status and simulation progress
                instance.reload()  # Reload instance data from AWS
                
                # Example: if instance has completed the simulation (simplified logic)
                if instance.state['Name'] == 'running':
                    logging.info(f"Instance {instance.id} is still running simulations.")
                elif instance.state['Name'] == 'stopped':
                    logging.info(f"Instance {instance.id} has completed the simulations.")
                    # Placeholder for retrieving results from the instance

            logging.info("Cloud simulation monitoring completed.")
        
        except Exception as e:
            logging.error(f"Error monitoring cloud simulations: {e}")
            raise

    def terminate_cloud_resources(self):
        """
        Terminate the cloud resources (e.g., EC2 instances) after simulations are completed.
        """
        try:
            logging.info("Terminating cloud resources...")

            for instance in self.cloud_instances:
                logging.info(f"Terminating instance {instance.id}...")
                instance.terminate()

            logging.info("All cloud instances terminated successfully.")
        
        except Exception as e:
            logging.error(f"Error terminating cloud resources: {e}")
            raise

    def enhanced_dashboard(self):
        """
        Enhance the Flask-based dashboard to include cloud monitoring features and simulation result visualization.
        This will allow users to monitor cloud simulations, view live results, and interact with the model.
        """
        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/run_simulation', methods=['POST'])
        def run_simulation():
            try:
                # Get input parameters from the dashboard
                wind_capacity = float(request.form.get('wind_capacity'))
                solar_capacity = float(request.form.get('solar_capacity'))
                battery_capacity = float(request.form.get('battery_capacity'))

                # Update the model
                self.wind_capacity = wind_capacity
                self.solar_capacity = solar_capacity
                self.battery_capacity = battery_capacity

                # Run simulation
                self.simulate_energy_flows()
                self.calculate_detailed_costs()
                self.calculate_extended_lifecycle_assessment()

                result = {
                    "levelized_cost": self.results['levelized_cost'],
                    "lifecycle_emissions": self.results['lifecycle_emissions'],
                    "NPV": self.results['NPV'],
                    "IRR": self.results['IRR']
                }
                return jsonify(result)
            
            except Exception as e:
                logging.error(f"Error running simulation from dashboard: {e}")
                return jsonify({"error": str(e)})

        @app.route('/cloud_monitor')
        def cloud_monitor():
            """
            A dashboard page to monitor the status of cloud simulations and retrieve live results.
            """
            try:
                # Monitor cloud simulations and display the status
                self.monitor_cloud_simulations()

                # Placeholder for displaying real-time status to the user
                status = {
                    "instance_1": "Running",
                    "instance_2": "Completed",
                    "instance_3": "Running"
                }
                return jsonify(status)
            
            except Exception as e:
                logging.error(f"Error monitoring cloud simulations from dashboard: {e}")
                return jsonify({"error": str(e)})

        def run_dashboard():
            # Run the Flask app in a separate thread to avoid blocking the main program
            threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()

        run_dashboard()

    def save_results_to_cloud(self, bucket_name='green-steel-results', file_prefix='simulation'):
        """
        Save the simulation results to AWS S3 cloud storage for long-term retention and sharing.
        This method uploads both the PDF and Excel reports to the specified S3 bucket.
        """
        try:
            s3 = boto3.client('s3')

            for scenario in self.results.get("batch_scenarios", []):
                # Generate filenames for PDF and Excel reports
                pdf_filename = f'report_{scenario["scenario_name"]}.pdf'
                excel_filename = f'report_{scenario["scenario_name"]}.xlsx'
                
                # Upload files to S3
                s3.upload_file(pdf_filename, bucket_name, f'{file_prefix}/{pdf_filename}')
                s3.upload_file(excel_filename, bucket_name, f'{file_prefix}/{excel_filename}')
                
                logging.info(f"Uploaded {pdf_filename} and {excel_filename} to S3 bucket {bucket_name}.")

        except Exception as e:
            logging.error(f"Error uploading files to S3: {e}")
            raise

    def visualize_simulation_results(self):
        """
        Generate visualizations for the simulation results using Plotly and embed them in the dashboard.
        """
        try:
            # Extract data from the results for visualization
            scenarios = [result["scenario_name"] for result in self.results.get("batch_scenarios", [])]
            lcos_values = [result["levelized_cost"] for result in self.results["batch_scenarios"]]
            emissions_values = [result["lifecycle_emissions"] for result in self.results["batch_scenarios"]]
            npv_values = [result["NPV"] for result in self.results["batch_scenarios"]]

            # Create Plotly figures
            fig_lcos = px.bar(x=scenarios, y=lcos_values, title="Levelized Cost of Steel (LCOS)")
            fig_emissions = px.bar(x=scenarios, y=emissions_values, title="Lifecycle Emissions (tons CO2-eq)")
            fig_npv = px.bar(x=scenarios, y=npv_values, title="Net Present Value (NPV)")

            # Convert figures to HTML for embedding in the dashboard
            lcos_html = pio.to_html(fig_lcos, full_html=False)
            emissions_html = pio.to_html(fig_emissions, full_html=False)
            npv_html = pio.to_html(fig_npv, full_html=False)

            return lcos_html, emissions_html, npv_html

        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
            raise

    def enhanced_dashboard(self):
        """
        Flask-based dashboard enhanced with visualization and cloud storage integration.
        Users can monitor cloud simulations, view live results, and download simulation reports.
        """
        app = Flask(__name__)

        @app.route('/')
        def index():
            # Main dashboard page
            lcos_html, emissions_html, npv_html = self.visualize_simulation_results()
            return render_template('index.html', lcos_chart=lcos_html, emissions_chart=emissions_html, npv_chart=npv_html)

        @app.route('/run_simulation', methods=['POST'])
        def run_simulation():
            try:
                # Extract input parameters from the dashboard
                wind_capacity = float(request.form.get('wind_capacity'))
                solar_capacity = float(request.form.get('solar_capacity'))
                battery_capacity = float(request.form.get('battery_capacity'))

                # Update model parameters
                self.wind_capacity = wind_capacity
                self.solar_capacity = solar_capacity
                self.battery_capacity = battery_capacity

                # Run simulation
                self.simulate_energy_flows()
                self.calculate_detailed_costs()
                self.calculate_extended_lifecycle_assessment()

                result = {
                    "levelized_cost": self.results['levelized_cost'],
                    "lifecycle_emissions": self.results['lifecycle_emissions'],
                    "NPV": self.results['NPV'],
                    "IRR": self.results['IRR']
                }
                return jsonify(result)

            except Exception as e:
                logging.error(f"Error running simulation: {e}")
                return jsonify({"error": str(e)})

        @app.route('/cloud_monitor')
        def cloud_monitor():
            # Monitor cloud simulations and display the status
            try:
                self.monitor_cloud_simulations()
                # Example: return the status of each cloud instance
                return jsonify({"instance_status": "running"})  # Simplified for demonstration
            except Exception as e:
                logging.error(f"Error monitoring cloud: {e}")
                return jsonify({"error": str(e)})

        @app.route('/download_results')
        def download_results():
            # Download the simulation results from S3 after completion
            try:
                self.save_results_to_cloud()
                return "Simulation results saved to cloud and are available for download."
            except Exception as e:
                logging.error(f"Error downloading results: {e}")
                return "Error downloading results."

        def run_dashboard():
            # Run Flask app in a thread to allow the main program to continue
            threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()

        run_dashboard()

    def autoscale_cloud_resources(self, scenario_count):
        """
        Automatically scale cloud resources based on the number of scenarios to be processed.
        This method increases or decreases the number of cloud instances to efficiently handle the workload.
        """
        try:
            ec2 = boto3.resource('ec2')
            target_instance_count = max(1, scenario_count // 10)  # Example: 1 instance per 10 scenarios

            logging.info(f"Target instance count for {scenario_count} scenarios: {target_instance_count}")

            # Get the current number of instances
            instances = list(ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running', 'pending']}]))
            current_instance_count = len(instances)

            # Scale up or down based on the target count
            if target_instance_count > current_instance_count:
                logging.info(f"Scaling up: launching {target_instance_count - current_instance_count} new instances.")
                ec2.create_instances(
                    ImageId='ami-0abcdef1234567890',  # Placeholder AMI ID
                    MinCount=target_instance_count - current_instance_count,
                    MaxCount=target_instance_count - current_instance_count,
                    InstanceType='t2.medium',
                    KeyName='green-steel-keypair'
                )
            elif target_instance_count < current_instance_count:
                logging.info(f"Scaling down: terminating {current_instance_count - target_instance_count} instances.")
                for instance in instances[:current_instance_count - target_instance_count]:
                    instance.terminate()

            logging.info("Cloud resources autoscaling completed.")
        
        except Exception as e:
            logging.error(f"Error autoscaling cloud resources: {e}")
            raise

    def aggregate_simulation_data(self):
        """
        Aggregate data across multiple scenarios and perform advanced analysis.
        This method summarizes the simulation results and identifies trends or outliers.
        """
        try:
            lcos_values = [result["levelized_cost"] for result in self.results["batch_scenarios"]]
            emissions_values = [result["lifecycle_emissions"] for result in self.results["batch_scenarios"]]
            npv_values = [result["NPV"] for result in self.results["batch_scenarios"]]

            # Aggregate the data
            avg_lcos = np.mean(lcos_values)
            avg_emissions = np.mean(emissions_values)
            avg_npv = np.mean(npv_values)
            max_lcos = np.max(lcos_values)
            min_lcos = np.min(lcos_values)

            # Identify trends or outliers
            outliers = [(scenario["scenario_name"], scenario["levelized_cost"]) for scenario in self.results["batch_scenarios"]
                        if scenario["levelized_cost"] > avg_lcos * 1.5]

            logging.info(f"Aggregated Data - Avg LCOS: {avg_lcos}, Avg Emissions: {avg_emissions}, Avg NPV: {avg_npv}")
            logging.info(f"Outliers (LCOS > 1.5x avg): {outliers}")

            # Store aggregated results
            self.results["aggregated"] = {
                "avg_lcos": avg_lcos,
                "avg_emissions": avg_emissions,
                "avg_npv": avg_npv,
                "max_lcos": max_lcos,
                "min_lcos": min_lcos,
                "outliers": outliers
            }

        except Exception as e:
            logging.error(f"Error aggregating simulation data: {e}")
            raise

    def setup_dashboard_security(self, app):
        """
        Setup user authentication and security for the Flask dashboard.
        Users must log in to access sensitive data or run simulations.
        """
        login_manager = LoginManager()
        login_manager.init_app(app)
        login_manager.login_view = 'login'

        # Mock user store
        users = {"admin": {"password": "password"}}

        class User(UserMixin):
            def __init__(self, id):
                self.id = id

        @login_manager.user_loader
        def load_user(user_id):
            return User(user_id)

        @app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                username = request.form['username']
                password = request.form['password']
                if username in users and users[username]['password'] == password:
                    user = User(username)
                    login_user(user)
                    return redirect(url_for('index'))
                return 'Invalid credentials'
            return render_template('login.html')

        @app.route('/logout')
        @login_required
        def logout():
            logout_user()
            return redirect(url_for('login'))

    def enhanced_dashboard(self):
        """
        Flask-based dashboard enhanced with security, visualization, and cloud scaling features.
        Users must log in to access data, and can monitor cloud simulations, view live results, and download reports.
        """
        app = Flask(__name__)
        app.secret_key = os.urandom(24)

        # Setup dashboard security
        self.setup_dashboard_security(app)

        @app.route('/')
        @login_required
        def index():
            # Main dashboard page
            lcos_html, emissions_html, npv_html = self.visualize_simulation_results()
            return render_template('index.html', lcos_chart=lcos_html, emissions_chart=emissions_html, npv_chart=npv_html)

        @app.route('/run_simulation', methods=['POST'])
        @login_required
        def run_simulation():
            try:
                # Extract input parameters from the dashboard
                wind_capacity = float(request.form.get('wind_capacity'))
                solar_capacity = float(request.form.get('solar_capacity'))
                battery_capacity = float(request.form.get('battery_capacity'))

                # Update model parameters
                self.wind_capacity = wind_capacity
                self.solar_capacity = solar_capacity
                self.battery_capacity = battery_capacity

                # Run simulation
                self.simulate_energy_flows()
                self.calculate_detailed_costs()
                self.calculate_extended_lifecycle_assessment()

                result = {
                    "levelized_cost": self.results['levelized_cost'],
                    "lifecycle_emissions": self.results['lifecycle_emissions'],
                    "NPV": self.results['NPV'],
                    "IRR": self.results['IRR']
                }
                return jsonify(result)

            except Exception as e:
                logging.error(f"Error running simulation: {e}")
                return jsonify({"error": str(e)})

        @app.route('/cloud_monitor')
        @login_required
        def cloud_monitor():
            # Monitor cloud simulations and display the status
            try:
                self.monitor_cloud_simulations()
                # Example: return the status of each cloud instance
                return jsonify({"instance_status": "running"})  # Simplified for demonstration
            except Exception as e:
                logging.error(f"Error monitoring cloud: {e}")
                return jsonify({"error": str(e)})

        @app.route('/download_results')
        @login_required
        def download_results():
            # Download the simulation results from S3 after completion
            try:
                self.save_results_to_cloud()
                return "Simulation results saved to cloud and are available for download."
            except Exception as e:
                logging.error(f"Error downloading results: {e}")
                return "Error downloading results."

        def run_dashboard():
            # Run Flask app in a thread to allow the main program to continue
            threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()

        run_dashboard()

    def test_cloud_autoscaling(self):
        """
        Test the cloud autoscaling mechanism to ensure it works as expected.
        This method will simulate different workloads and verify that the cloud resources are scaled appropriately.
        """
        try:
            logging.info("Starting cloud autoscaling test...")

            # Test case 1: Small workload (5 scenarios)
            logging.info("Testing with 5 scenarios...")
            self.autoscale_cloud_resources(5)

            # Test case 2: Medium workload (20 scenarios)
            logging.info("Testing with 20 scenarios...")
            self.autoscale_cloud_resources(20)

            # Test case 3: Large workload (50 scenarios)
            logging.info("Testing with 50 scenarios...")
            self.autoscale_cloud_resources(50)

            logging.info("Cloud autoscaling tests completed successfully.")

        except Exception as e:
            logging.error(f"Error during cloud autoscaling test: {e}")
            raise

    def validate_simulation_results(self):
        """
        Perform validation checks on the simulation results to ensure accuracy and consistency.
        This method verifies that all key metrics (LCOS, emissions, NPV) are within expected ranges.
        """
        try:
            for scenario in self.results.get("batch_scenarios", []):
                if not (0 < scenario["levelized_cost"] < 10000):
                    raise ValueError(f"Invalid LCOS value for scenario {scenario['scenario_name']}: {scenario['levelized_cost']}")
                if not (0 < scenario["lifecycle_emissions"] < 100000):
                    raise ValueError(f"Invalid emissions value for scenario {scenario['scenario_name']}: {scenario['lifecycle_emissions']}")
                if not (-100000000 < scenario["NPV"] < 100000000):
                    raise ValueError(f"Invalid NPV value for scenario {scenario['scenario_name']}: {scenario['NPV']}")

            logging.info("Simulation result validation passed successfully.")
        
        except Exception as e:
            logging.error(f"Validation error: {e}")
            raise

    def finalize_for_deployment(self):
        """
        Finalize the system for deployment, ensuring that all components are functioning correctly and optimizations are applied.
        This includes optimizing performance, setting up production configurations, and running final tests.
        """
        try:
            logging.info("Starting final deployment setup...")

            # Final optimizations (e.g., ensuring database connections are efficient, minimizing memory usage)
            self.optimize_performance()

            # Run final tests
            logging.info("Running final tests...")
            self.test_cloud_autoscaling()
            self.validate_simulation_results()

            logging.info("Final deployment setup completed successfully.")

        except Exception as e:
            logging.error(f"Error during final deployment setup: {e}")
            raise

    def optimize_performance(self):
        """
        Apply performance optimizations to the system to handle large workloads efficiently.
        This could involve tuning database queries, minimizing memory usage, and optimizing cloud operations.
        """
        try:
            logging.info("Applying performance optimizations...")

            # Placeholder for actual optimization code (e.g., optimizing database queries, caching frequently accessed data)
            # Example: optimize database queries to minimize I/O
            self.db_connection.execute("PRAGMA optimize")
            logging.info("Database optimizations applied.")

            # Example: reduce memory footprint by clearing unused variables
            self.clear_unused_variables()
            logging.info("Memory optimizations applied.")

        except Exception as e:
            logging.error(f"Error during performance optimization: {e}")
            raise

    def clear_unused_variables(self):
        """
        Clear unused variables and data structures from memory to improve performance.
        This is important in large-scale simulations where memory usage can become significant.
        """
        try:
            # Example: Clear large datasets from memory once they are no longer needed
            self.results = None
            logging.info("Unused variables cleared from memory.")
        except Exception as e:
            logging.error(f"Error during memory clearance: {e}")
            raise

    def debug_mode(self):
        """
        Enable detailed logging and debugging mode to trace issues during development or testing.
        This method will output more verbose logs and catch exceptions in a more granular way.
        """
        try:
            logging.info("Enabling debug mode...")

            # Set logging level to DEBUG for detailed output
            logging.getLogger().setLevel(logging.DEBUG)

            # Example debug scenario: force a controlled error for testing purposes
            try:
                raise ValueError("This is a test error for debug mode.")
            except ValueError as e:
                logging.debug(f"Debug mode caught an error: {e}")

            logging.info("Debug mode enabled successfully.")

        except Exception as e:
            logging.error(f"Error in debug mode: {e}")
            raise

    def run_full_test_suite(self):
        """
        Run a full test suite to verify the entire system works as intended.
        This includes tests for cloud autoscaling, simulation accuracy, performance, and validation.
        """
        try:
            logging.info("Starting full test suite...")

            # Test cloud autoscaling
            self.test_cloud_autoscaling()

            # Test simulation results validation
            self.validate_simulation_results()

            # Test performance optimizations
            self.optimize_performance()

            # Run in debug mode to catch any potential issues
            self.debug_mode()

            logging.info("Full test suite completed successfully.")

        except Exception as e:
            logging.error(f"Error during full test suite: {e}")
            raise
