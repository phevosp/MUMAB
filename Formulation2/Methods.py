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

class Arm:
    """
    Arm class for multi-armed bandit problem

    Attributes:
        id:             int, unique identifier for arm
        true_mean:      float, true mean of the arm
        num_pulls:      int, number of times the arm has been pulled
        total_reward:   int, total reward accumulated from pulling arm. Only notes reward from single pull of arm
        estimated_mean: int, estimated mean of the arm
        conf_radius:    int, confidence radius of the arm
        ucb:            int, upper confidence bound of the arm
        function:       function, function used to compute multiplicative benefit of multiple agents sampling the same arm

    Methods:
        get_reward:     returns random reward for pulling the arm
        pull:           updates the arm's attributes after simulating a pull of the arm
        reset:         resets the arm's attributes
    """
    def __init__(self, id):
        self.id             :int   = id
        self.true_mean      :float = random.random() * 0.5 + 0.25
        self.num_pulls      :int   = 0
        self.total_reward   :int   = 0
        self.estimated_mean :int   = 0
        self.conf_radius    :int   = 0
        self.ucb            :int   = 0

        # We use 2+(id%10) as the base of the log
        # All these functions are concave and increasing with f(0) = 0 and f(1) = 1
        log_base                   = 2 + (id%10)
        self.function              = lambda x : (np.emath.logn(log_base, 0.05*x + 1/(log_base)) + 1) / (np.emath.logn(log_base, 0.05 + 1/(log_base)) + 1)

    def get_reward(self):
        return np.random.normal(loc = self.true_mean, scale = 0.06)
    
    def pull(self, time, num_agents):
        # print("Sampling Arm {} at time {}\n".format(self.id, time))
        single_reward = self.get_reward()
        reward = self.function(num_agents) * single_reward
        self.num_pulls += 1
        self.total_reward += single_reward
        self.estimated_mean = self.total_reward / self.num_pulls
        self.conf_radius = np.sqrt(2 * np.log(time) / self.num_pulls)
        self.ucb = self.estimated_mean + self.conf_radius
        # print("Single Reward: {}\n".format(single_reward))
        return reward
    
    def reset(self):
        self.num_pulls = 0
        self.total_reward = 0
        self.estimated_mean = 0
        self.conf_radius = 0
        self.ucb = 0

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
    
    def pull(self, time, agents, agent_ids):
        # print("Sampling Arm {} at time {}\n".format(self.id, time))
        single_reward = self.get_reward()
        reward = self.function(len(agent_ids)) * single_reward
        for a_id in agent_ids:
            agents[a_id].num_pulls_dict[self.id] += 1
            agents[a_id].total_reward_dict[self.id] += single_reward
            agents[a_id].estimated_mean_dict[self.id] = agents[a_id].total_reward_dict[self.id] / agents[a_id].num_pulls_dict[self.id]
            agents[a_id].conf_radius_dict[self.id] = np.sqrt(2 * np.log(time) / agents[a_id].num_pulls_dict[self.id])
            agents[a_id].ucb_dict[self.id] = agents[a_id].estimated_mean_dict[self.id] + agents[a_id].conf_radius_dict[self.id]

        # print("Single Reward: {}\n".format(single_reward))
        return reward
    
class Agent:
    """
    Agent class for multi-armed bandit problem.

    Attributes:
        id:             int, unique identifier for agent
        current_node:   dict (networkX.node), current node at which the agent is located

    Methods:
        move:           moves the agent to the inputted node
    """
    def __init__(self, id, node):
        # Agent attributes
        self.id           :int  = id
        self.current_node :dict = node

    def move(self, new_node):
        self.current_node = new_node

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
        self.num_pulls_dict      = {i: 0 for i in G}
        self.total_reward_dict   = {i: 0 for i in G}
        self.estimated_mean_dict = {}
        self.conf_radius_dict    = {}
        self.ucb_dict            = {}

    def move(self, new_node):
        self.current_node = new_node

def initialize(agents):
    """
        Initializes the approximations for each vertex by having each agent run DFS until all the vertices are visited at least once.
    """
    curr_time = 1
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

    # Sample current arms as well
    curr_time + 1
    for agent in agents:
        # Add current vertex to arm_dict
        if agent.current_node['arm'] not in arm_dict:
            arm_dict[agent.current_node['arm']] = 1
        else:
            arm_dict[agent.current_node['arm']] += 1

    rew_per_turn.append(0)
    for arm in arm_dict:
        rew_per_turn[-1] += arm.pull(curr_time, arm_dict[arm])
    return curr_time, rew_per_turn

def initialize_indv(G, agents):
    """
        Initializes the approximations for each vertex by having each agent run DFS until it has visited all the vertices.
    """
    curr_time = 1
    # Keep track of reward per turn
    rew_per_turn = []

    # Initialize list of visited nodes for each agent
    visited    = {agent.id:[False for node in G] for agent in agents}
    for agent in agents:
        print(agent.id, agent.current_node['id'])
        visited[agent.id][agent.current_node['id']] = True

    # Initialize list of previous nodes for each agent
    prev_nodes = {agent.id:{} for agent in agents}

    # While there is some agent that has not visited all nodes
    while not all([sum(visited[agent.id]) == len(G) for agent in agents]):
        curr_time += 1
        # arm_dict will be the collection of arms that are visited at the current time
        # arm_dict[arm] = list of agent.ids that are sampling the arm
        arm_dict = {}
        for agent in agents:
            # Add current agent to arm_dict
            if agent.current_node['arm'] not in arm_dict:
                arm_dict[agent.current_node['arm']] = [agent.id]
            else:
                arm_dict[agent.current_node['arm']].append(agent.id)

            # Move to first neighbor that has not been visited
            moved = False
            for neighbor_id in nx.all_neighbors(G, agent.current_node['id']):
                if not visited[agent.id][neighbor_id]:
                    prev_nodes[agent.id][neighbor_id] = agent.current_node['id']
                    agent.move(G.nodes[neighbor_id])

                    # Immediately mark as visited, though we haven't yet sampled reward
                    visited[agent.id][agent.current_node['id']] = True
                    moved                             = True
                    break
            
            # If we didn't move forward then move backwards
            if not moved:  
                prev_node_id = prev_nodes[agent.id][agent.current_node['id']]
                agent.move(G.nodes[prev_node_id])

        rew_per_turn.append(0)
        for arm in arm_dict:
            rew_per_turn[-1] += arm.pull(curr_time, agents, arm_dict[arm])

    
    # Sample current arms as well
    for agent in agents:
        if agent.current_node['arm'] not in arm_dict:
            arm_dict[agent.current_node['arm']] = [agent.id]
        else:
            arm_dict[agent.current_node['arm']].append(agent.id)

    rew_per_turn.append(0)
    for arm in arm_dict:
        rew_per_turn[-1] += arm.pull(curr_time, agents, arm_dict[arm])

    # Double check all vertices have been visited
    for agent in agents:
        assert(sorted(agent.estimated_mean_dict.keys()) == sorted(G.nodes()))

    return curr_time, rew_per_turn

def optimal_distribution(arm_list, theoretical = False):
    """
        Calculates the optimal distribution of agents over the arms.
        If theoretical == False, uses the current UCB estimates of each arm.
        If theoretical == True, uses the true means of each arm.
    """
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    store_vars = {}
    # This is the number of agents selecting each arm
    for arm in arm_list:
        # This is the number of agents selecting each arm (call it x)
        store_vars[f"x_{arm.id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = M, name = f"x_{arm.id}")
        # This is 0.05*x_{} + 1/(log_base)
        temp1                                = m.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0.0, name = f"0.05*x_{arm.id}+1/(log_base)")
        # This is np.emath.logn(log_base, 0.05*x + 1/(log_base))
        temp2                                = m.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = f"(np.emath.logn(log_base,0.05*x_{arm.id}+1/(log_base)))")
        # This is f(x) = (np.emath.logn(log_base, 0.05*x + 1/(log_base)) + 1) / (np.emath.logn(log_base, 0.05 + 1/(log_base)) + 1)
        store_vars[f"f(x_{arm.id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{arm.id})")

        # We use 2+(id%10) as the base of the log
        log_base = 2+(arm.id%10)

        # Add constraints
        m.addConstr(temp1 == 0.05 * store_vars[f"x_{arm.id}"] + 1/log_base, name = f"constr1_x_{arm.id}")
        m.addGenConstrLogA(temp1, temp2, log_base)
        m.addConstr(store_vars[f"f(x_{arm.id})"] == (temp2+1)/(np.emath.logn(log_base, 0.05 + 1/log_base) + 1), name = f"constr2_x_{arm.id}")

    # Constraint that we can only pick M times
    m.addConstr(sum([store_vars[f"x_{arm.id}"] for arm in arm_list]) == M)
    if not theoretical:
        m.setObjective(sum([arm.ucb * store_vars[f"f(x_{arm.id})"] for arm in arm_list]), gp.GRB.MAXIMIZE)
    if theoretical:
        m.setObjective(sum([arm.true_mean * store_vars[f"f(x_{arm.id})"] for arm in arm_list]), gp.GRB.MAXIMIZE)

    m.optimize()
    if m.status == gp.GRB.INFEASIBLE:
        m.computeIIS()
        m.write("model.lp")
        m.write("model.ilp")

    store_values = m.getAttr("X", store_vars)
    return store_values, m.getObjective().getValue()

def episode(agents, curr_time, type):
    """
        Runs one episode of the algorithm. 
        Optimal distribution is computed using UCB estimates of each arm.
        Agents move to their assigned destination node, and sample that node until the baseline_arm has its samples doubled.
        If type == 'original': baseline_arm is the arm with the least number of pulls
        If type == 'median':   baseline_arm is the arm with the median number of pulls
        If type == 'max':      baseline_arm is the arm with the most number of pulls
    """
    # Keep track of reward per turn
    rew_per_turn = []

    # Compute optimal distribution
    distribution, _ = optimal_distribution([G.nodes[node]['arm'] for node in G])

    # Create list of sampled nodes
    # If a node is to be sampled n times then it appears n times in the list
    # The number of times the node is sampled is of course x_node = x_G.nodes[node]['arm'].id
    sampled_nodes = []
    for node in G:
        for times in range(round(distribution[f"x_{G.nodes[node]['arm'].id}"])):
            sampled_nodes.append(node)
    # f.write("Sampled Nodes: {}\n".format(sampled_nodes))
    # print("Distribution:", distribution)
    # print("Sampled Nodes:", sampled_nodes)

    # Note number of pulls of baseline
    sorted_by_pulls = sorted(sampled_nodes, key = lambda x : G.nodes[x]['arm'].num_pulls)
    if type == 'original':
        baseline_arm = sorted_by_pulls[0]
    elif type == 'median':
        baseline_arm = sorted_by_pulls[len(sampled_nodes) // 2]
    elif type == 'max':
        baseline_arm = sorted_by_pulls[-1]

    baseline_pulls = G.nodes[baseline_arm]['arm'].num_pulls
    # f.write("Baseline Arm: {}\n".format(baseline_arm))

    # Note maximum ucb value for edge
    max_ucb = max([G.nodes[node]['arm'].ucb for node in sampled_nodes])
    
    # Initialize edge weights given UCB estimate of each arm.
    # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
    G_directed = nx.DiGraph(G)
    for (u, v) in G.edges():
        G_directed.edges[u, v]["weight"] = max_ucb - G.nodes[v]['arm'].ucb
        G_directed.edges[v, u]["weight"] = max_ucb - G.nodes[u]['arm'].ucb

    # For each agent and optimal arm pair compute shortest path to create weights for bipartite graph
    # sp_dict is indexed by (agent_id, node_i) and stores a tuple (path length, actual path)
    # where path is the shortest path between the current node of the agent and the destination node
    sp_dict = {}
    for agent in agents:
        # Compute single source shortest path
        shortest_path        = nx.shortest_path(G_directed, source = agent.current_node['id'], weight = "weight")
        shortest_path_length = nx.shortest_path_length(G_directed, source = agent.current_node['id'], weight = "weight")
        # And then add path to shortest path dictionary for all destination nodes
        for i, dest_node in enumerate(sampled_nodes):
            sp_dict[(agent.id, f"{dest_node}_{i}")] = (shortest_path_length[dest_node], shortest_path[dest_node])

    # Create bipartite graph
    B = nx.Graph()
    B.add_nodes_from([('agent', agent.id) for agent in agents])
    B.add_nodes_from([(f'node_{i}', node) for i, node in enumerate(sampled_nodes)])
    for (agent_id, dest_node_str) in sp_dict:
        dest_node = int(dest_node_str.split('_')[0])
        index    = int(dest_node_str.split('_')[1])
        B.add_edge(('agent', agent_id), (f'node_{index}', dest_node), weight = sp_dict[(agent_id, dest_node_str)][0])
    assignments = bipartite.minimum_weight_full_matching(B, top_nodes = [('agent', agent.id) for agent in agents], weight = "weight")


    # Create list paths where paths[i] is the path for agent i
    paths = [[] for _ in agents]
    for agent in agents:
        (node_name, dest_node) = assignments[('agent', agent.id)]
        index  = int(node_name.split('_')[1])
        paths[agent.id] = sp_dict[(agent.id, f"{dest_node}_{index}")][1]

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
            # If we have more path left, move to next node
            if i < len(paths[agent.id]):
                agent.move(G.nodes[paths[agent.id][i]])

            # Then add current vertex to arm_dict
            if agent.current_node['arm'] not in arm_dict:
                arm_dict[agent.current_node['arm']] = 1
            else:
                arm_dict[agent.current_node['arm']] += 1

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
    if G.nodes[baseline_arm]['arm'] not in arm_dict:
        assert(curr_time == T)

    while G.nodes[baseline_arm]['arm'].num_pulls < 2 * baseline_pulls and curr_time < T:
        curr_time += 1
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
        paths[agent.id]      = nx.shortest_path(G_directed, source = agent.current_node['id'], target = opt_arm, weight = "weight")

    # baseline_pulls gives the number of pulls of the arm with the least number of pulls by its respective agent
    # baseline_agent gives the agent who has pulled their respective arm the fewest number of times
    # baseline_arm gives the arm with the least number of pulls
    baseline_pulls = min([agent.num_pulls_dict[paths[agent.id][-1]] for agent in agents])
    baseline_agent = [agent for agent in agents if agent.num_pulls_dict[paths[agent.id][-1]] == baseline_pulls][0]
    baseline_arm   = paths[baseline_agent.id][-1] 

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
                arm_dict[agent.current_node['arm']] = [agent.id]
            else:
                arm_dict[agent.current_node['arm']].append(agent.id)

            # If we have more path left, move to next node
            if i < len(paths[agent.id]):
                agent.move(G.nodes[paths[agent.id][i]])

        rew_per_turn.append(0)
        for arm in arm_dict:
            rew_per_turn[-1] += arm.pull(curr_time, agents, arm_dict[arm])

    # We update arm_dict
    arm_dict = {}
    for agent in agents:
        # Add current vertex to arm_dict
        if agent.current_node['arm'] not in arm_dict:
            arm_dict[agent.current_node['arm']] = [agent.id]
        else:
            arm_dict[agent.current_node['arm']].append(agent.id)

    # We sample until num_pulls of baseline_arm doubles
    if baseline_agent.current_node['id'] !=baseline_arm:
        assert(curr_time == T)

    while baseline_agent.num_pulls_dict[baseline_arm] < 2 * baseline_pulls and curr_time < T:
        curr_time += 1
        rew_per_turn.append(0)
        for arm in arm_dict:
            rew_per_turn[-1] += arm.pull(curr_time, agents, arm_dict[arm])

    return curr_time, rew_per_turn

def run(type):
    # Reset arms
    for i in G:
        G.nodes[i]['arm'].reset()
    
    # Initialize agents and assign vertex
    agents   = [Agent(i, G.nodes[random.randint(0, K-1)]) for i in range(M)]

    # Begin Algorithm
    # After the return from each function call, 
    # curr_time is the last time step that was run
    # And each agent has yet to sample from their current vertex
    curr_time = 0
    curr_ep   = 0
    curr_time, reward_per_turn = initialize(agents)

    while curr_time < T:
        curr_ep   += 1
        print("Episode {}".format(curr_ep))
        # f.write("Starting Episode {}, Current time is {}\n".format(curr_ep, curr_time))
        curr_time, new_rewards = episode(agents, curr_time, type)
        reward_per_turn += new_rewards

    # Get results
    for node in G.nodes():
        arm = G.nodes[node]['arm']
        f.write('\nArm ' + str(arm.id))
        f.write("\nTrue Mean: " + str(arm.true_mean))
        f.write("\nEstimated Mean: " +  str(arm.estimated_mean))
        f.write("\nUCB Value: " + str(arm.ucb))
        f.write("\nNum Pulls: " + str(arm.num_pulls))

    return reward_per_turn, curr_time

def run_indv(G):
    """
        Runs the individual learner algorithm on the inputted graph G for T time steps.
    """
    # Initialize graph using individual arm objects
    for i in G:
        G.nodes[i]['arm'] = Arm_indv(i)
        # print(G.nodes[i]['arm'].true_mean)
        G.nodes[i]['id']  = i
        G.nodes[i]['prev_node'] = G.nodes[i]
    
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
    for agent in agents:
        f.write('--------------------------------------------------')
        f.write('\nAgent ' + str(agent.id))
        for node in G.nodes():
            arm = G.nodes[node]['arm']
            f.write('\nArm ' + str(arm.id))
            f.write("\nTrue Mean: " + str(arm.true_mean))
            f.write("\nEstimated Mean: " +  str(agent.estimated_mean_dict[arm.id]))
            f.write("\nUCB Value: " + str(agent.ucb_dict[arm.id]))
            f.write("\nNum Pulls: " + str(agent.num_pulls_dict[arm.id]))

    return reward_per_turn, curr_time
"""
---------------------------------------------------------------------------------------------------------------------------------------
    Executable Code
--------------------------------------------------------------------------------------------------------------------------------------- 
"""
# Note that variations in the results come from an unstable maximum weight matching algorithm in the 'episode' function
cumulative_regrets = {}
type_list = ['original', 'median', 'max', "individual"]
names     = ["G-combUCB", "G-combUCB-median", "G-combUCB-max", "Multi-G-UCB"]

# Problem Parameters
T = 15000
K = 500
M = 20
p = 0.05
num_trials = 10

# Create Stochastic Setting
#---------------------------------------------------#
# Initialize connected graph
G = nx.erdos_renyi_graph(K, p, seed = 0)
tries = 0
while not nx.is_connected(G) and tries < 10:
    G = nx.erdos_renyi_graph(K, p, seed = 0 + tries)
    tries += 1
assert(nx.is_connected(G))

# Assign each vertex an associated arm
for i in G:
    G.nodes[i]['arm'] = Arm(i)
    # print(G.nodes[i]['arm'].true_mean)
    G.nodes[i]['id']  = i
    G.nodes[i]['prev_node'] = G.nodes[i]

# Visualize graph
sns.set_theme()
plt.clf()
nx.draw(G, with_labels = True)
plt.savefig("state_graph.png")

# Get theoretical max_per_turn
_, max_per_turn = optimal_distribution([G.nodes[node]['arm'] for node in G.nodes()], theoretical = True)

# Run algorithm num_times for each algorithmic type (min, median, max)
for type in type_list:
    if not os.path.exists(type):
        os.makedirs(type)

    cumulative_regrets[type] = []
    for trial in range(num_trials):
    # Initialize output file and results directory
        if not os.path.exists(type + "/trial_" + str(trial)):
            os.makedirs(type + "/trial_" + str(trial))
        
        # Open results file
        f = open(type + "/trial_" + str(trial) + "/{}.txt".format(sys.argv[1]), "w")

        if type != 'individual':
            reward_per_turn, curr_time = run(type)
        else:
            reward_per_turn, curr_time = run_indv(G)

        print('Total Time: ' + str(curr_time))
        f.write('\n----------------------------------------------------\n')
        f.write('Total Time: ' + str(curr_time))
        f.write("\nNet Reward: " + str(sum(reward_per_turn)))
        f.write("\nTheoretical Expected Max: " + str(T * max_per_turn))
        f.write("\n")
        f.close()

        # Calculate regret
        cum_regret = np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn))
        cumulative_regrets[type].append(cum_regret)

        # # Plot cumulative reward
        assert(len(reward_per_turn) == curr_time == T)
        plt.clf()
        plt.plot(range(T), np.cumsum(reward_per_turn), label = 'Observed')
        plt.plot(range(T), [max_per_turn * i for i in range(1, T+1)], label = 'Theoretical Max')
        plt.xlabel("Time")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative reward as a function of time")
        plt.legend()
        plt.savefig(type + "/trial_" + str(trial) + "/cumulative_reward.png")

        # # Plot Cumulative Regret
        plt.clf()
        plt.plot(range(T), np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)))
        plt.xlabel("Time")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative regret as a function of time")
        plt.savefig(type + "/trial_" + str(trial) + "/cumulative_regret.png")

        # # Plot Average Regret
        plt.clf()
        plt.plot(range(T), np.divide(np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)), range(1, T+1)))
        plt.xlabel("Time")
        plt.ylabel("Average Regret")
        plt.title("Average regret as a function of time")
        plt.savefig(type + "/trial_" + str(trial) + "/av_regret.png")


    # # Plot Cumulative Regret Averaged over all trials of current type
    av_cum_regret = np.mean(cumulative_regrets[type], axis = 0)

    plt.clf()
    for i in range(num_trials):
        plt.plot(range(T), cumulative_regrets[type][i], alpha = 0.4, color= 'grey')

    plt.plot(range(T), av_cum_regret, alpha = 0.7, color='orange')
    plt.xlabel("Time")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative regret as a function of time")
    plt.savefig(type + "/av_cumulative_regret.png")

    # # Plot Average Regret Averaged over all trials of current type
    plt.clf()
    for i in range(num_trials):
        plt.plot(range(T), np.divide(cumulative_regrets[type][i], range(1, T+1)), alpha = 0.4, color= 'grey')

    plt.plot(range(T), np.divide(av_cum_regret, range(1, T+1)), alpha = 0.7, color='orange')
    plt.xlabel("Time")
    plt.ylabel("Average Regret")
    plt.title("Average regret as a function of time")
    plt.savefig(type + "/av_average_regret.png")

    np.save(type + "/cumulative_regrets.npy", cumulative_regrets[type])


# # Plot Mean Regret for different algorithm types
plt.clf()
palette = sns.color_palette()
for i, type in enumerate(type_list):
    plt.plot(range(T), np.mean(cumulative_regrets[type], axis = 0), alpha = 0.9, color= palette[i], label = names[i])

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.title("Cumulative regret as a function of time")
plt.savefig("av_cumulative_regret_comparison.png")

# # Plot Average Regret for different algorithm types
plt.clf()
for i, type in enumerate(type_list):
    plt.plot(range(T), np.divide(np.mean(cumulative_regrets[type], axis = 0), range(1, T+1)), alpha = 0.9, color=palette[i], label = names[i])

plt.xlabel("Time")
plt.ylabel("Average Regret")
plt.legend()
plt.title("Average regret as a function of time")
plt.savefig("av_average_regret_comparison.png")