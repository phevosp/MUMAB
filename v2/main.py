import argparse
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import random
import networkx as nx
from networkx import bipartite
import sys
import os
import json
import time
import gurobipy as gp

import MUMAB.objects as mobj
from MUMAB.algorithms.Manager import Manager, plot_function_regrets

# Dictionary of implemented algorithms
alg_names = {
    'indv' : "Multi-G-UCB",
    'median' : "G-combUCB-median",
    'max' : "G-combUCB-max",
    'original' : "G-combUCB"
}

def load_params():
    parser = argparse.ArgumentParser(description='MUMAB hyper parameters')
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--p', type=float, default=0.05)
    parser.add_argument('--num_trials', type=int, default=10)    
    parser.add_argument('--function_types', nargs='+', default = ['concave'], choices=['concave', 'collision', 'more_concave', 'linear', 'constant'])
    parser.add_argument('--output_dirs', nargs= '+')
    parser.add_argument('--alg_types', nargs='+', default=['original'], choices=list(alg_names.keys()))
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    params = parser.parse_args()

    # Generate default output directory
    if params.output_dirs is None:
        params.output_dir = f"output/{params.T}-{params.K}-{params.M}/"
        params.output_dirs = [f"{params.output_dir}{func}/" for func in params.function_types]

    # Create directories
    for dir in params.output_dirs:
        try:
            os.makedirs(dir)
            print(f"Directory {dir} created successfully.")
        except FileExistsError:
            pass

    params.alg_names = [alg_names[type] for type in params.alg_types]
    return params


def initialize_graph(params):
    # To-do: Potentially, extend to different graph initializations 
    # Generate graph based on number of arms and probabilities
    G = nx.erdos_renyi_graph(params.K, params.p, seed = 0)
    tries = 0
    while not nx.is_connected(G) and tries < 10:
        G = nx.erdos_renyi_graph(params.K, params.p, seed = tries)
        tries += 1
    assert(nx.is_connected(G))
    return G


def setup_graph_interaction(G, function_type, params):
    # Assign each vertex an associated arm
    for i in G:
        G.nodes[i]['arm'] = mobj.Arm(i, mobj.MultiAgentInteraction.getFunction(i, function_type, params))
        G.nodes[i]['id']  = i
        G.nodes[i]['prev_node'] = G.nodes[i]

    return G


def main():
    params = load_params()

    G_ = initialize_graph(params)
    params.graph_diameter = nx.diameter(G_)

    regret_results = {}
    for output_dir, ftype in zip(params.output_dirs, params.function_types):
        print(f'================================================================================Evaluating {ftype} Performance ================================================================================')
        G = G_.copy()
        setup_graph_interaction(G, ftype, params)
        manager = Manager(params, G)
        regret_results = manager.evaluate_algs(output_dir, regret_results, ftype)

    plot_function_regrets(params, regret_results)

if __name__ == '__main__':
    main()


