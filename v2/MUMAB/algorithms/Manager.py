from .utils.OptimalDistribution import optimal_distribution
from .utils.Plotter import Plotter as plt
from .MAB import MAB, getMAB
import os
import sys

import numpy as np
from tqdm import tqdm

class Manager():
    def __init__(self, params, G):
        self.params = params
        self.G = G
        self.cumulative_regrets = {}
        self.T = params.T
    
    def evaluate_type(self, max_per_turn, alg_type, alg_name):
        self.cumulative_regrets[alg_type] = []
        mab_alg = getMAB(alg_type, self.G, self.params) 

        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating {alg_name}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        for trial in range(self.params.num_trials):
            print(f'Running Trial {trial}')
            trial_output_dir = f"{self.params.output_dir}{alg_type}/trial_{trial}"
            if not os.path.exists(trial_output_dir):
                os.makedirs(trial_output_dir)
            
            # Open results file
            f = open(f"{trial_output_dir}/results.txt", "w")
            reward_per_turn, curr_time = mab_alg.run(f)
 
            # print('Total Time: ' + str(curr_time))
            f.write('\n----------------------------------------------------\n')
            f.write('Total Time: ' + str(curr_time))
            f.write("\nNet Reward: " + str(sum(reward_per_turn)))
            f.write("\nTheoretical Expected Max: " + str(self.T * max_per_turn))
            f.write("\n")
            f.close()

            # Calculate regret
            cum_regret = np.subtract([max_per_turn * i for i in range(1, self.T+1)], np.cumsum(reward_per_turn))
            self.cumulative_regrets[alg_type].append(cum_regret)

            plt.plot_cumulative_reward(reward_per_turn, max_per_turn, trial_output_dir, self.T)
            plt.plot_cumulative_regret(reward_per_turn, max_per_turn, trial_output_dir, self.T)
            plt.plot_average_regret(reward_per_turn, max_per_turn, trial_output_dir, self.T)            
        return np.mean(self.cumulative_regrets[alg_type], axis = 0)

    def evaluate_algs(self):
        # Get theoretical max_per_turn
        _, max_per_turn = optimal_distribution([self.G.nodes[node]['arm'] for node in self.G.nodes()], self.params.M, theoretical = True)
        # Run algorithm num_times for each algorithmic type (min, median, max)
        for name, type in zip(self.params.alg_names, self.params.alg_types):
            av_cum_regret = self.evaluate_type(max_per_turn, type, name)
            output_dir = f"{self.params.output_dir}{type}"
            plt.plot_cumulative_regret_total(self.cumulative_regrets[type], av_cum_regret, output_dir, self.T)
            plt.plot_average_regret_total(self.cumulative_regrets[type], av_cum_regret, output_dir, self.T)

        plt.plot_algs_mean_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, self.params.output_dir, self.T)
        plt.plot_algs_mean_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, self.params.output_dir, self.T, log_scaled=True)
        plt.plot_algs_avg_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, self.params.output_dir, self.T)
        plt.plot_algs_avg_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, self.params.output_dir, self.T, log_scaled=True)
