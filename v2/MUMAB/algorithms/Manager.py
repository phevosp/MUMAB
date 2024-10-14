from .utils.OptimalDistribution import optimal_distribution, compare_dist
from .utils.Plotter import Plotter as plt
from .MAB import MAB, getMAB
from collections import Counter
import os
import sys

import numpy as np
from tqdm import tqdm


class Manager:
    def __init__(self, params, G, Gindv):
        self.params = params
        self.G = G
        self.Gindv = Gindv
        self.cumulative_regrets = {}
        self.T = params.T

    def _evaluate_type(
        self,
        max_reward_per_turn,
        max_regret_per_turn,
        alg_type,
        alg_name,
        best_alloc,
        output_dir,
        thresh=0.015,
    ):
        # Goal: Given an algorithm type and a graph, evaluate the algorithm on the graph for num_trials trials
        # Note: max_per_reward_turn is the maximum reward possible per-turn.
        # Note: max_regret_per_turn is the maximum regret possible per-turn.
        mab_alg = getMAB(alg_type, self.G, self.Gindv, self.params)

        print(
            f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating {alg_name}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        regrets = np.zeros((self.params.num_trials, self.params.T))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for trial in range(self.params.num_trials):
            print(f"Running Trial {trial}")


            # Open results file
            regret_trial = mab_alg.run(max_reward_per_turn)
            regrets[trial] = regret_trial

        output_file = f"{output_dir}{alg_name}.csv"
        np.savetxt(output_file, regrets, delimiter=",")

    def evaluate_algs(self, output_dir, regret, ftype):

        # Goal: Evaluate all algorithms for a particular function type
        for n in self.G.nodes():
            print(self.G.nodes[n]["arm"])

        # Get theoretical max_per_turn and calculate max regret
        best_dist, _ = optimal_distribution(
            [self.G.nodes[node]["arm"] for node in self.G.nodes()],
            self.params,
            theoretical=True,
            minimize=False,
            debug=True,
            output_dir=output_dir,
        )
        worst_dist, _ = optimal_distribution(
            [self.G.nodes[node]["arm"] for node in self.G.nodes()],
            self.params,
            theoretical=True,
            minimize=True,
            debug=True,
            output_dir=output_dir,
        )
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
            max_reward_per_turn += (
                self.G.nodes[key]["arm"].interaction.function(best_dict[key])
                * self.G.nodes[key]["arm"].true_mean
            )
        for key in worst_dict:
            min_reward_per_turn += (
                self.G.nodes[key]["arm"].interaction.function(worst_dict[key])
                * self.G.nodes[key]["arm"].true_mean
            )

        max_regret_per_turn = max_reward_per_turn - min_reward_per_turn
        print(
            f"Maximum Per Turn: {max_reward_per_turn}, \nMinimum Per Turn: {min_reward_per_turn}, \nMax Regret: {max_regret_per_turn}"
        )

        ## list of optimal arms
        optimal_dist = [key for key in best_dict]

        # Run algorithm num_times for each algorithmic type (min, median, max)
        for name, type in zip(self.params.alg_names, self.params.alg_types):
            
            # Call evaluate_type on specific algorithm
            self._evaluate_type(
                max_reward_per_turn,
                max_regret_per_turn,
                type,
                name,
                optimal_dist,
                output_dir,
            )