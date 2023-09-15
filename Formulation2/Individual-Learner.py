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

class Arm_indv:
    """
    Arm class for multi-armed bandit problem

    Attributes:
        id:             int, unique identifier for arm
        true_mean:      float, true mean of the arm
        function:       function, function used to compute multiplicative benefit of multiple agents sampling the same arm

    Methods:
        get_reward:     returns random reward for pulling the arm
        pull:           updates each agents' view of the arm after simulating a pull of the arm
    """
    def __init__(self, id):
        self.id             :int   = id
        self.true_mean      :float = random.random() * 0.5 + 0.25

        # We use 2+(id%10) as the base of the log
        # All these functions are concave and increasing with f(0) = 0 and f(1) = 1
        log_base                   = 2 + (id%10)
        self.function              = lambda x : (np.emath.logn(log_base, 0.05*x + 1/(log_base)) + 1) / (np.emath.logn(log_base, 0.05 + 1/(log_base)) + 1)

    def get_reward(self):
        return np.random.normal(loc = self.true_mean, scale = 0.06)
    
    def pull(self, time, agents):
        # print("Sampling Arm {} at time {}\n".format(self.id, time))
        single_reward = self.get_reward()
        reward = self.function(len(agents)) * single_reward
        for agent in agents:
            agent.num_pulls_dict[self.id] += 1
            agent.total_reward_dict[self.id] += single_reward
            agent.estimated_mean_dict[self.id] = agent.total_reward_dict[self.id] / agent.num_pulls_dict[self.id]
            agent.conf_radius_dict[self.id] = np.sqrt(2 * np.log(time) / agent.num_pulls_dict[self.id])
            agent.ucb_dict[self.id] = agent.estimated_mean_dict[self.id] + agent.conf_radius_dict[self.id]

        # print("Single Reward: {}\n".format(single_reward))
        return reward


    
class Agent_indv:
    """
    Agent class for multi-armed bandit problem.

    Attributes:
        id:                   int, unique identifier for agent
        current_node:         dict (networkX.node), current node at which the agent is located
        num_pulls_dict:       dict (arm_id : num_pulls), dictionary of arms that the agent has pulled and the number of times it has pulled each arm
        total_reward_dict:    dict (arm_id : total_reward), dictionary of arms that the agent has pulled and the total reward it has received from each arm
        estimated_mean_dict:  dict (arm_id : estimated_mean), dictionary of arms that the agent has pulled and the estimated mean of each arm
        conf_radius_dict:     dict (arm_id : conf_radius), dictionary of arms that the agent has pulled and the confidence radius of each arm
        ucb_dict:             dict (arm_id : ucb), dictionary of arms that the agent has pulled and the upper confidence bound of each arm

    Methods:
        move:           moves the agent to the inputted node
    """
    def __init__(self, id, node):
        # Agent attributes
        self.id           :int  = id
        self.current_node :dict = node
        self.num_pulls_dict      = {}
        self.total_reward_dict   = {}
        self.estimated_mean_dict = {}
        self.conf_radius_dict    = {}
        self.ucb_dict            = {}

    def move(self, new_node):
        self.current_node = new_node


def initialize_indv(G, agents):
    """
        Initializes the approximations for each vertex by having each agent run DFS until all the vertices are visited at least once.
    """
    curr_time = 0
    # Keep track of reward per turn
    rew_per_turn = []

    # Initialize list of visited nodes
    visited    = [False for node in G]
    for agent in agents:
        visited[agent.current_node['id']] = True

    while not all(visited):
        print(sum(visited))
        curr_time += 1
        # arm_dict will be the collection of arms that are visited at the current time
        arm_dict = {}
        for agent in agents:
            # Add current vertex to arm_dict
            if agent.current_node['arm'] not in arm_dict:
                arm_dict[agent.current_node['arm']] = 1
            else:
                arm_dict[agent.current_node['arm']] += 1

            # Move to first neighbor that has not been visited
            moved = False
            for neighbor_id in nx.all_neighbors(G, agent.current_node['id']):
                if not visited[neighbor_id]:
                    G.nodes[neighbor_id]['prev_node'] = agent.current_node
                    agent.move(G.nodes[neighbor_id])

                    # Immediately mark as visited, though we haven't yet sampled reward
                    visited[agent.current_node['id']] = True
                    moved                             = True
                    break
            
            # If we didn't move forward then move backwards
            if not moved:    
                agent.move(agent.current_node['prev_node'])

        rew_per_turn.append(0)
        for arm in arm_dict:
            rew_per_turn[-1] += arm.pull(curr_time, arm_dict[arm])

    return curr_time, rew_per_turn

def episode_indv(G, agents, curr_time):
    """
        Runs one episode of the algorithm. 
        Each agent chooses the arm with highest UCB value.
        Agents move to their assigned destination node, and sample that node until one agent communicates to all others that they should stop.
    """
    # Keep track of reward per turn
    rew_per_turn = []

    # For each agent, find shortest path to arm with max UCB value
    paths = [[] for _ in agents]
    for agent in agents:
        # Find arm with max UCB value
        opt_arm = max(agent.ucb_dict, key = agent.ucb_dict.get)
        max_ucb = agent.ucb_dict[opt_arm]

        # Find shortest path to arm with max UCB values
        # Initialize edge weights given UCB estimate of each arm.
        # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
        # This way, edges leading to nodes with higher ucb have lower weight
        G_directed = nx.DiGraph(G)
        for (u, v) in G.edges():
            G_directed.edges[u, v]["weight"] = max_ucb - agent.ucb_dict[G.nodes[v]['arm'].id]
            G_directed.edges[v, u]["weight"] = max_ucb - agent.ucb_dict[G.nodes[u]['arm'].id]
    
        # Compute single source shortest path and add to dictionary
        shortest_path        = nx.shortest_path(G_directed, source = agent.current_node['id'], weight = "weight")
        paths[agent.id]      = shortest_path

    # baseline_pulls gives the number of pulls of the arm with the least number of pulls by its respective agent
    # baseline_agent gives the agent who has pulled their respective arm the fewest number of times
    # baseline_arm gives the arm with the least number of pulls
    baseline_pulls = min([agent.num_pulls_dict[paths[agent.id][-1]] for agent in agents])
    baseline_agent = [agent for agent in agents if agent.num_pulls[paths[agent.id][-1]] == baseline_pulls][0]
    baseline_arm   = paths[baseline_agent][-1] 

    # f.write("Paths: {}\n".format(paths))
    # Move agents along paths, if agents have reached the end of their journey then they stay at desination node
    max_path_length = max([len(path) for path in paths])
    i = 0
    while i < max_path_length and curr_time < T:
        curr_time += 1
        i += 1
        # arm_dict will be the set of arms that are visited at the current time
        arm_dict = {}
        for agent in agents:
            # Add current vertex to arm_dict
            if agent.current_node['arm'] not in arm_dict:
                arm_dict[agent.current_node['arm']] = 1
            else:
                arm_dict[agent.current_node['arm']] += 1

            # If we have more path left, move to next node
            if i < len(paths[agent.id]):
                agent.move(G.nodes[paths[agent.id][i]])

        rew_per_turn.append(0)
        for arm in arm_dict:
            rew_per_turn[-1] += arm.pull(curr_time, arm_dict[arm])

    # We update arm_dict
    arm_dict = {}
    for agent in agents:
        # Add current vertex to arm_dict
        if agent.current_node['arm'] not in arm_dict:
            arm_dict[agent.current_node['arm']] = 1
        else:
            arm_dict[agent.current_node['arm']] += 1

    # We sample until num_pulls of baseline_arm doubles
    if baseline_agent.current_node['id'] !=baseline_arm:
        assert(curr_time == T)

    while baseline_agent.num_pulls_dict[baseline_arm] < 2 * baseline_pulls and curr_time < T:
        curr_time += 1
        rew_per_turn.append(0)
        for arm in arm_dict:
            rew_per_turn[-1] += arm.pull(curr_time, arm_dict[arm])

    return curr_time, rew_per_turn


"""
---------------------------------------------------------------------------------------------------------------------------------------
    Executable Code
--------------------------------------------------------------------------------------------------------------------------------------- 
"""

def run_individual(G, T, K, M, trial):
    """
        To be called from Methods.py

        Runs the individual learner algorithm on the inputted graph G for T time steps.

        Inputs:
            G:     networkX graph
            T:     int, time horizon
            K:     int, number of arms
            M:     int, number of agents
            trial: int, trial number

        Outputs:
            The reward_per_turn of the individual learner algorithm  
    """
    # Initialize graph using individual arm objects
    for i in G:
        G.nodes[i]['arm'] = Arm_indv(i)
        # print(G.nodes[i]['arm'].true_mean)
        G.nodes[i]['id']  = i
        G.nodes[i]['prev_node'] = G.nodes[i]

    # Initialize output file and results directory
    if not os.path.exists("individual/trial_" + str(trial)):
        os.makedirs("individual/trial_" + str(trial))
    
    # Open results file
    f = open("individual/trial_" + str(trial) + "/{}.txt".format(sys.argv[1]), "w")
    
    # Initialize agents and assign vertex
    agents   = [Agent_indv(i, G.nodes[random.randint(0, K-1)]) for i in range(M)]

    # Begin Algorithm
    # After the return from each function call, 
    # curr_time is the last time step that was run
    # And each agent has yet to sample from their current vertex
    curr_time = 0
    curr_ep   = 0
    curr_time, reward_per_turn = initialize_indv(G, agents)

    while curr_time < T:
        curr_ep   += 1
        print("Episode {}".format(curr_ep))
        # f.write("Starting Episode {}, Current time is {}\n".format(curr_ep, curr_time))
        curr_time, new_rewards = episode_indv(G, agents, curr_time)
        reward_per_turn += new_rewards

    # Get results
    net_reward = 0
    for node in G.nodes():
        arm = G.nodes[node]['arm']
        f.write('\nArm ' + str(arm.id))
        f.write("\nTrue Mean: " + str(arm.true_mean))
        f.write("\nEstimated Mean: " +  str(arm.estimated_mean))
        f.write("\nUCB Value: " + str(arm.ucb))
        f.write("\nNum Pulls: " + str(arm.num_pulls))
        net_reward += arm.total_reward

    return reward_per_turn