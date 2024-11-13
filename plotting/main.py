import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_params():
    parser = argparse.ArgumentParser(description='MUMAB hyper parameters')
    parser.add_argument('--data_folder', default="/home/anaveen/Documents/research_ws/MUMAB/output/3500000-50-4/constant", type=str)
    params = parser.parse_args()
    return params

def main():
    params = load_params()
    # List all CSV files
    csv_files = [f for f in os.listdir(params.data_folder) if f.endswith('.csv') and 'intervals' not in f]

    palette = sns.color_palette()
    
    for i, f in enumerate(csv_files):
        regrets = np.loadtxt(f"{params.data_folder}/{f}", delimiter=",", dtype=float, ndmin=2)
            
        cumulative_regrets = np.cumsum(regrets, axis=1)
        mean_fit = np.mean(cumulative_regrets, axis=0)
        min_fit = np.min(cumulative_regrets, axis=0)
        max_fit = np.max(cumulative_regrets, axis=0)
        T = regrets.shape[1]
        plt.plot(range(T), mean_fit, alpha = 0.9, color= palette[i], label = f[:-4])
        plt.fill_between(range(T), min_fit, max_fit,
                                color='gray', alpha=0.2)

        # time = np.arange(0, T)
        # for interval in transition_interval:
        #     plt.fill_between(time,0, mean_fit, where=(time >= interval[0]) & (time <= interval[1]), color = 'gray')

        
    plt.xlabel("Time")
    plt.ylabel("Cumulative Regret")
    plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.grid(True)
    plt.legend()
    plt.title("Cumulative regret as a function of time")

    plt.show()

if __name__ == '__main__':
    main()            
