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

# Dictionary of implemented algorithms
alg_names = {
    "simple": "Multi-G-UCB",
    "robust": "Robust-Multi-G-UCB",
    "indv": "Indv-Multi-G-UCB",
    "UCRL2": "Multi-UCRL2",
    "max": "Max-Multi-G-UCB",
}


def load_params():
    """
    Load hyperparameters from command line arguments
    """
    parser = argparse.ArgumentParser(description="MUMAB hyper parameters")
    parser.add_argument("--T", type=int, default=1000000)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument(
        "--function_types",
        nargs="+",
        default=["log"],
        choices=["log", "collision", "more_log", "linear", "constant", "power"],
    )
    parser.add_argument("--numer", nargs="+", default=[1])
    parser.add_argument("--denom", nargs="+", default=[2])
    parser.add_argument("--output_dirs", nargs="+")
    parser.add_argument(
        "--alg_types", nargs="+", default=["indv"], choices=list(alg_names.keys())
    )
    parser.add_argument("--normalized", type=bool, default=True)
    parser.add_argument("--agent_std_dev", nargs="+", type=float, default=None)
    parser.add_argument("--agent_bias", nargs="+", type=float, default=None)
    parser.add_argument("--agent_move_prob", nargs="+", type=float, default=None)
    parser.add_argument("--agent_sample_prob", nargs="+", type=float, default=None)
    parser.add_argument("--agent_move_alpha", nargs="+", type=float, default=None)
    parser.add_argument("--agent_sample_alpha", nargs="+", type=float, default=None)
    parser.add_argument("--agent_move_beta", nargs="+", type=float, default=None)
    parser.add_argument("--agent_sample_beta", nargs="+", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument(
        "--delta", type=float, default=0.01
    )  # confidence parameter of UCRL2
    parser.add_argument("--output_flag", type=str, default="")
    parser.add_argument("options", default=None, nargs=argparse.REMAINDER)
    params = parser.parse_args()

    # Create multiple power functions, one per numerator, denominator pair
    if "power" in params.function_types:
        try:
            assert len(params.numer) == len(params.denom)
        except:
            raise ValueError(
                "List of numerators and denominators must be of the same length"
            )

        params.function_types.remove("power")
        for numer, denom in zip(params.numer, params.denom):
            params.function_types.append(f"power_{numer}_{denom}")

    # Generate default output directory
    if params.output_dirs is None:
        params.output_dir = f"output/{params.output_flag}/"
        params.output_dirs = [
            f"{params.output_dir}{func}/" for func in params.function_types
        ]

    # Create directories
    for dir in params.output_dirs:
        try:
            os.makedirs(dir)
            print(f"Directory {dir} created successfully.")
        except FileExistsError:
            pass

    if params.agent_std_dev is None:
        params.agent_std_dev = np.zeros(
            params.M,
        )

    if params.agent_bias is None:
        params.agent_bias = np.zeros(
            params.M,
        )

    if params.agent_move_prob is None:
        params.agent_move_prob = np.ones(
            params.M,
        )

    if params.agent_sample_prob is None:
        params.agent_sample_prob = np.ones(
            params.M,
        )

    if params.agent_sample_alpha is None:
        params.agent_sample_alpha = params.M * [None]

    if params.agent_move_alpha is None:
        params.agent_move_alpha = params.M * [None]

    if params.agent_sample_beta is None:
        params.agent_sample_beta = params.M * [None]

    if params.agent_move_beta is None:
        params.agent_move_beta = params.M * [None]

    params.alg_names = [alg_names[type] for type in params.alg_types]
    return params


def initialize_graph(params):
    """
    Initialize a connected graph with K vertices and edge probability p
    """
    G = nx.erdos_renyi_graph(params.K, params.p, seed=0)
    tries = 0
    while not nx.is_connected(G) and tries < 10:
        G = nx.erdos_renyi_graph(params.K, params.p, seed=tries)
        tries += 1
    assert nx.is_connected(G)
    return G


def setup_graph_interaction(G, function_type, params):
    """
    Initialize the graph.
    For each vertex, create an arm object with the corresponding function type.
    Create also an individual arm type for use in the individual algorithm.
    """
    for i in G:
        G.nodes[i]["arm"] = mobj.Arm(
            i,
            mobj.MultiAgentInteraction.getFunction(i, function_type, params),
            params.K,
        )
        G.nodes[i]["id"] = i
        G.nodes[i]["prev_node"] = G.nodes[i]

    G_ = G.copy()
    for i in G:
        G_.nodes[i]["arm"] = mobj.ArmIndividual(G.nodes[i]["arm"], params.M)
    return G, G_


def main():
    """
    Main function to run the MUMAB algorithms
    """
    params = load_params()

    G_ = initialize_graph(params)
    params.graph_diameter = nx.diameter(G_)

    for output_dir, ftype in zip(params.output_dirs, params.function_types):
        print(
            f"================================================================================Evaluating {ftype} Performance ================================================================================"
        )
        G = G_.copy()
        G, Gindv = setup_graph_interaction(G, ftype, params)
        manager = Manager(params, G, Gindv)
        manager.evaluate_algs(output_dir)


if __name__ == "__main__":
    main()
