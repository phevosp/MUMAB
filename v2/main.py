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
from MUMAB.algorithms.Manager import Manager

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
    parser.add_argument('--function_type', type=str, choices=['concave', 'collision'], default='concave')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--alg_types', nargs='+', default=[], choices=list(alg_names.keys()))
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    params = parser.parse_args()

    if params.output_dir is None:
        params.output_dir = f"output/{params.function_type}_{params.T}-{params.K}-{params.M}/"

    try:
        os.mkdir(params.output_dir)
        print(f"Directory {params.output_dir} created successfully.")
    except FileExistsError:
        pass

    params.alg_names = [alg_names[type] for type in params.alg_types]
    return params


def setup_graph(params, function_type):
    G = nx.erdos_renyi_graph(params.K, params.p, seed = 0)
    tries = 0
    while not nx.is_connected(G) and tries < 10:
        G = nx.erdos_renyi_graph(params.K, params.p, seed = tries)
        tries += 1
    assert(nx.is_connected(G))

    # Assign each vertex an associated arm
    for i in G:
        G.nodes[i]['arm'] = mobj.Arm(i, mobj.MultiAgentInteraction.getFunction(i, function_type, params))
        G.nodes[i]['id']  = i
        G.nodes[i]['prev_node'] = G.nodes[i]

    return G


def main():
    params = load_params()

    G = setup_graph(params, params.function_type)

    manager = Manager(params, G)
    manager.evaluate_algs()


if __name__ == '__main__':
    main()


