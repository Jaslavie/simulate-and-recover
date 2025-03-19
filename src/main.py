# main script to run simulations and save results
import pandas as pd
from datetime import datetime
from src.simulation import Simulation
from utils.plot import plot_parameter_recovery, plot_bias_and_mse, display_summary_statistics

def main():
    simulation = Simulation()
    results = simulation.run_all_simulations()

    if not results.empty:
        print("Generating plots...")
        plot_parameter_recovery(results)
        summary = plot_bias_and_mse(results)
        display_summary_statistics(summary)
    else:
        print("No data to analyze")

if __name__ == "__main__":
    main()