from .utils.OptimalDistribution import optimal_distribution, compare_dist
from .utils.Plotter import Plotter as plt
from .MAB import MAB, getMAB
from collections import Counter
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
    
    def _evaluate_type(self, max_reward_per_turn, max_regret_per_turn, alg_type, alg_name, best_alloc, output_dir, thresh=0.015):
        # Goal: Given an algorithm type and a graph, evaluate the algorithm on the graph for num_trials trials
        # Note: max_per_reward_turn is the maximum reward possible per-turn.
        # Note: max_regret_per_turn is the maximum regret possible per-turn.
        self.cumulative_regrets[alg_type] = []
        mab_alg = getMAB(alg_type, self.G, self.params) 

        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating {alg_name}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        for trial in range(self.params.num_trials):
            print(f'Running Trial {trial}')
            trial_output_dir = f"{output_dir}{alg_type}/trial_{trial}"
            if not os.path.exists(trial_output_dir):
                os.makedirs(trial_output_dir)
            
            # Open results file
            f = open(f"{trial_output_dir}/results.txt", "w")
            reward_per_turn, curr_time, transition_intervals, episode_allocations = mab_alg.run(f)
            learned, ious = compare_dist(best_alloc, episode_allocations)
            f.write('\n----------------------------------------------------\n')
            f.write('Total Time: ' + str(curr_time))
            f.write("\nNet Reward: " + str(sum(reward_per_turn)))
            f.write("\nTheoretical Expected Max: " + str(self.T * max_reward_per_turn))
            f.write("\n Algorithm Learned: " + str(learned))
            f.write("\n")
            f.close()

            regret = max_reward_per_turn - np.array(reward_per_turn)
            learned = np.mean(regret[-int(0.05*self.params.T):]) < thresh

            print(f"\n ALGORITHM LEARNED: {learned}")
            # Calculate regret
            cum_regret = np.subtract(np.array([max_reward_per_turn * i for i in range(1, self.T+1)]), np.cumsum(reward_per_turn))
            # If normalized, divide the cumulative regret by the max_regret_per_turn
            if self.params.normalized:
                cum_regret = np.divide(cum_regret, max_regret_per_turn)
            self.cumulative_regrets[alg_type].append(cum_regret)

            plt.plot_cumulative_reward(reward_per_turn, max_reward_per_turn, trial_output_dir, self.T, self.params.normalized)
            plt.plot_cumulative_regret(reward_per_turn, max_reward_per_turn, transition_intervals, trial_output_dir, self.T, self.params.normalized)
            plt.plot_average_regret(reward_per_turn, max_reward_per_turn, trial_output_dir, self.T, self.params.normalized)    
            plt.plot_transition_regret_per_episode_cost(reward_per_turn, max_reward_per_turn, transition_intervals, trial_output_dir, self.T)
            plt.plot_iou(ious, trial_output_dir)        
        return np.mean(self.cumulative_regrets[alg_type], axis = 0)

    def evaluate_algs(self, output_dir, regret, ftype):
        
        # Goal: Evaluate all algorithms for a particular function type
        for n in self.G.nodes():
            print(self.G.nodes[n]['arm'])

        # Get theoretical max_per_turn and calculate max regret
        best_dist, _ = optimal_distribution([self.G.nodes[node]['arm'] for node in self.G.nodes()], self.params, theoretical = True, minimize=False, debug=True, output_dir = output_dir)
        worst_dist, _ = optimal_distribution([self.G.nodes[node]['arm'] for node in self.G.nodes()], self.params, theoretical = True, minimize=True, debug=True, output_dir = output_dir)
        # Print best and worst distributions
        for dist, name in zip([best_dist, worst_dist], ["Best", "Worst"]):
            sampled_nodes = []
            for node in self.G:
                for _ in range(round(dist[f"x_{self.G.nodes[node]['arm'].id}"])):
                    sampled_nodes.append(node)
            
            if name == "Best":
                best_dict = Counter(sampled_nodes)
            else:
                worst_dict = Counter(sampled_nodes)
            print(f"In {name} distribution, we sample nodes:", sampled_nodes)

        # Given sampled nodes, calculate theoretical max and mins (using Gurobi reward was giving errors)

        max_reward_per_turn, min_reward_per_turn = 0, 0
        for key in best_dict:
            max_reward_per_turn += self.G.nodes[key]['arm'].interaction.function(best_dict[key]) * self.G.nodes[key]['arm'].true_mean
        for key in worst_dict:
            min_reward_per_turn += self.G.nodes[key]['arm'].interaction.function(worst_dict[key]) * self.G.nodes[key]['arm'].true_mean

        max_regret_per_turn = max_reward_per_turn - min_reward_per_turn
        print(f"Maximum Per Turn: {max_reward_per_turn}, \nMinimum Per Turn: {min_reward_per_turn}, \nMax Regret: {max_regret_per_turn}")

        ## list of optimal arms
        optimal_dist = [key for key in best_dict]


        # Run algorithm num_times for each algorithmic type (min, median, max)
        for name, type in zip(self.params.alg_names, self.params.alg_types):
            # Initialize sub-dictionary the first time called on a particular algorithm type 
            if not type in regret: regret[type] = {}

            # Call evaluate_type on specific algorithm
            regret[type][ftype] = self._evaluate_type(max_reward_per_turn, max_regret_per_turn, type, name, optimal_dist, output_dir)
            output_dir_type = f"{output_dir}{type}"
            plt.plot_cumulative_regret_total(self.cumulative_regrets[type], regret[type][ftype], output_dir_type, self.T, self.params.normalized)
            plt.plot_average_regret_total(self.cumulative_regrets[type], regret[type][ftype], output_dir_type, self.T, self.params.normalized)

        plt.plot_algs_cum_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, output_dir, self.T, self.params.normalized)
        plt.plot_algs_cum_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, output_dir, self.T, self.params.normalized, log_scaled=True)
        plt.plot_algs_avg_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, output_dir, self.T, self.params.normalized)
        plt.plot_algs_avg_regret(self.cumulative_regrets, self.params.alg_names, self.params.alg_types, output_dir, self.T, self.params.normalized, log_scaled=True)
        return regret
    
def plot_function_regrets(params, regret):
    for alg_name, alg_type in zip(params.alg_names, params.alg_types):
        plt.plot_algs_avg_regret_ftypes(regret[alg_type], params.function_types, alg_type, alg_name, params.T, params.output_dir, params.normalized)
        plt.plot_algs_cum_regret_ftypes(regret[alg_type], params.function_types, alg_type, alg_name, params.T, params.output_dir, params.normalized)