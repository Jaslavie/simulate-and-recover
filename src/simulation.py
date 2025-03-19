# simulate experiment with diffusion model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from src.diffusion_model import DiffusionModel

class Simulation:
    def __init__(self):
        self.diffusion_model = DiffusionModel()
        self.sample_sizes = [10, 40, 4000]
        self.output_dir = "results"
        self.n_simulations = 1000 # as defined in instructions
    
    def run_simulation(self, n_trials):
        # input data of the model
        a_true, v_true, t0_true = self.diffusion_model.randomly_select_parameters()
        acc, rt = self.diffusion_model.simulate_trial_data(a_true, v_true, t0_true, n_trials)

        # summary stats (output of model)
        R_obs = np.mean(acc)
        M_obs = np.mean(rt)
        V_obs = np.var(rt)

        # check validity of observations
        if R_obs <= 0.5 or R_obs >=1.0 or V_obs <= 0:
            return None

        # recover parameters from observation (i.e. test the inverse function)
        a_rec, v_rec, t0_rec = self.diffusion_model.inverse(R_obs, M_obs, V_obs)
        # calculate bias and squared error (bias average to 0, squared error decreases as N increases)
        # test this in the test cases to ensure the conditions are met
        a_bias = a_rec - a_true
        v_bias = v_rec - v_true
        t0_bias = t0_rec - t0_true
        
        a_squared_error = a_bias ** 2
        v_squared_error = v_bias ** 2
        t0_squared_error = t0_bias ** 2

        return n_trials, a_true, v_true, t0_true, a_rec, v_rec, t0_rec, a_bias, v_bias, t0_bias, a_squared_error, v_squared_error, t0_squared_error

    def run_multiple_simulations(self, n_trials, n_simulations):
        # run multiple simulate-and-recover simulations with given sample sizes
        # return dataframe of results
        # column names
        columns = ['n_trials', 
                    'a_true', 'v_true', 't0_true', 
                    'a_rec', 'v_rec', 't0_rec', 
                    'a_bias', 'v_bias', 't0_bias', 
                    'a_squared_error', 'v_squared_error', 't0_squared_error']
        results = []
        successful_count = 0 
        # run simulation loop
        for i in range(n_simulations):
            result = self.run_simulation(n_trials)
            if result is not None:
                results.append(result)
                successful_count += 1
            # print progress
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{n_simulations} simulations completed")
        # return dataframe of results
        if results:
            df = pd.DataFrame(results, columns=columns)
            return df
        else:
            return pd.DataFrame(columns=columns)
    def run_all_simulations(self):
        # run all simulations with all sample sizes
        results = []
        for n_trials in self.sample_sizes:
            df = self.run_multiple_simulations(n_trials, self.n_simulations)

            # save results to a csv file for easier analysis
            if not df.empty:
                df.to_csv(f"{self.output_dir}/results_n{n_trials}.csv")
                results.append(df)

                # print summary statistics
                print(f"Simulation results for N={n_trials}:")
                print(f"  Mean a bias: {df['a_bias'].mean():.4f}")
                print(f"  Mean v bias: {df['v_bias'].mean():.4f}")
                print(f"  Mean t0 bias: {df['t0_bias'].mean():.4f}")
                print(f"  Mean a squared error: {df['a_squared_error'].mean():.4f}")
                print(f"  Mean v squared error: {df['v_squared_error'].mean():.4f}")
                print(f"  Mean t0 squared error: {df['t0_squared_error'].mean():.4f}")
        
        # combine results
        if results:
            combined_df = pd.concat(results)
            combined_df.to_csv(f"{self.output_dir}/results_all.csv")
            return combined_df
        else:
            return pd.DataFrame()
