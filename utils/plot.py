#### GENERATED BY CLAUDE ####
# used to visualize the results of the simulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_parameter_recovery(results_df, output_dir="results"):
    """
    Plot parameter recovery (true vs. recovered).
    
    Parameters:
    - results_df: DataFrame containing simulation results
    - output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parameters to plot
    parameters = [('a', 'Boundary Separation'), 
                  ('v', 'Drift Rate'), 
                  ('t0', 'Nondecision Time')]
    
    # Plot for each sample size
    for n_trials in results_df['n_trials'].unique():
        subset = results_df[results_df['n_trials'] == n_trials]
        
        for param_code, param_name in parameters:
            plt.figure(figsize=(8, 6))
            
            plt.scatter(subset[f'{param_code}_true'], subset[f'{param_code}_rec'], 
                      alpha=0.5, edgecolor='none')
            
            # Add identity line
            min_val = min(subset[f'{param_code}_true'].min(), subset[f'{param_code}_rec'].min())
            max_val = max(subset[f'{param_code}_true'].max(), subset[f'{param_code}_rec'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')
            
            plt.xlabel(f'True {param_name}')
            plt.ylabel(f'Recovered {param_name}')
            plt.title(f'Parameter Recovery: {param_name} (N={n_trials})')
            
            plt.savefig(f"{output_dir}/{param_code}_recovery_n{n_trials}.png")
            plt.close()

def plot_bias_and_mse(results_df, output_dir="results"):
    """
    Plot bias and mean squared error by sample size.
    
    Parameters:
    - results_df: DataFrame containing simulation results
    - output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parameters to plot
    parameters = [('a', 'Boundary Separation'), 
                  ('v', 'Drift Rate'), 
                  ('t0', 'Nondecision Time')]
    
    # Calculate summary statistics by sample size
    summary = results_df.groupby('n_trials').agg({
        'a_bias': 'mean',
        'v_bias': 'mean',
        't0_bias': 'mean',
        'a_squared_error': 'mean',
        'v_squared_error': 'mean',
        't0_squared_error': 'mean'
    }).reset_index()
    
    # Sort by sample size
    summary = summary.sort_values('n_trials')
    
    # Plot bias
    plt.figure(figsize=(10, 6))
    
    for param_code, param_name in parameters:
        plt.plot(summary['n_trials'], summary[f'{param_code}_bias'], 
               'o-', label=f'{param_name}')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Parameter Bias')
    plt.title('Parameter Bias by Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_dir}/parameter_bias.png")
    plt.close()
    
    # Plot MSE
    plt.figure(figsize=(10, 6))
    
    for param_code, param_name in parameters:
        plt.plot(summary['n_trials'], summary[f'{param_code}_squared_error'], 
               'o-', label=f'{param_name}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Mean Squared Error')
    plt.title('Parameter Mean Squared Error by Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_dir}/parameter_mse.png")
    plt.close()
    
    # Save summary statistics
    summary.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    
    return summary

def display_summary_statistics(results_df):
    """
    Display summary statistics for the simulation results.
    
    Parameters:
    - results_df: DataFrame containing simulation results
    """
    # Calculate summary statistics by sample size
    summary = results_df.groupby('n_trials').agg({
        'a_bias': ['mean', 'std'],
        'v_bias': ['mean', 'std'],
        't0_bias': ['mean', 'std'],
        'a_squared_error': 'mean',
        'v_squared_error': 'mean',
        't0_squared_error': 'mean'
    })
    
    print("\nSummary Statistics:")
    print("===================")
    print(summary)
    
    return summary