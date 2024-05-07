import matplotlib.pyplot as plt 
import numpy as np
import random
import networkx as nx
from networkx import bipartite
import sys
import os
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
    """
    def __init__(self, id):
        self.id             :int   = id
        self.true_mean      :float = random.random() * 0.5 + 0.25
        self.num_pulls      :int   = 0
        self.total_reward   :int   = 0
        self.estimated_mean :int   = 0
        self.conf_radius    :int   = 0
        self.ucb            :int   = 0
        
    def get_reward(self):
        return np.random.normal(loc = self.true_mean, scale = 0.06)
    
    def pull(self, time):
        # f.write("Sampling Arm {} at time {}\n".format(self.id, time))
        reward = self.get_reward()
        self.num_pulls += 1
        self.total_reward += reward
        self.estimated_mean = self.total_reward / self.num_pulls
        self.conf_radius = np.sqrt(2 * np.log(time) / self.num_pulls)
        self.ucb = self.estimated_mean + self.conf_radius
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


def initialize():
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
        # arm_set will be the collection of arms that are visited at the current time
        arm_set = set()
        for agent in agents:
            # Add current vertex to arm_set
            arm_set.add(agent.current_node['arm'])

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
        for arm in arm_set:
            rew_per_turn[-1] += arm.pull(curr_time)

    return curr_time, rew_per_turn

def episode(curr_time):
    """
        Runs one episode of the algorithm. 
        Each agent moves to assigned destination node (one to each of the M arms with highest UCB).
        Agents sample their assigned destination node until at least one arm has been had its samples doubled.
    """
    # Keep track of reward per turn
    rew_per_turn = []

    # Compute optimal arms
    opt_nodes = sorted(list(G), reverse = True, key = lambda x : G.nodes[x]['arm'].ucb)[:len(agents)]
    # f.write("Optimal nodes: {}\n".format(opt_nodes))

    # Note number of pulls of arm with minimum number of pulls
    arm_with_min_pulls = sorted(opt_nodes, key = lambda x : G.nodes[x]['arm'].num_pulls)[0]
    min_pulls = G.nodes[arm_with_min_pulls]['arm'].num_pulls
    # f.write("Arm with min pulls: {}\n".format(arm_with_min_pulls))

    # Initialize edge weights given UCB estimate of each arm.
    # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
    G_directed = nx.DiGraph(G)
    for (u, v) in G.edges():
        G_directed.edges[u, v]["weight"] = G.nodes[opt_nodes[0]]['arm'].ucb - G.nodes[v]['arm'].ucb
        G_directed.edges[v, u]["weight"] = G.nodes[opt_nodes[0]]['arm'].ucb - G.nodes[u]['arm'].ucb

    # For each agent and optimal arm pair compute shortest path to create weights for bipartite graph
    # sp_dict is indexed by (agent_id, node) and stores a tuple (path length, actual path)
    # where path is the shortest path between the current node of the agent and the destination node
    sp_dict = {}
    for agent in agents:
        # Compute single source shortest path
        shortest_path        = nx.shortest_path(G_directed, source = agent.current_node['id'], weight = "weight")
        shortest_path_length = nx.shortest_path_length(G_directed, source = agent.current_node['id'], weight = "weight")
        # And then add path to shortest path dictionary for all destination nodes
        for dest_node in opt_nodes:
            sp_dict[(agent.id, dest_node)] = (shortest_path_length[dest_node], shortest_path[dest_node])

    # Create bipartite graph
    B = nx.Graph()
    B.add_nodes_from([('agent', agent.id) for agent in agents])
    B.add_nodes_from([('node', node) for node in opt_nodes])
    for (agent_id, dest_node) in sp_dict:
        B.add_edge(('agent', agent_id), ('node', dest_node), weight = sp_dict[(agent_id, dest_node)][0])
    assignments = bipartite.minimum_weight_full_matching(B, top_nodes = [('agent', agent.id) for agent in agents], weight = "weight")
    
    # Create list paths where paths[i] is the path for agent i
    paths = [[] for _ in agents]
    for agent in agents:
        (_, dest_node) = assignments[('agent', agent.id)]
        paths[agent.id]= sp_dict[(agent.id, dest_node)][1]

    # f.write("Paths: {}\n".format(paths))
    # Move agents along paths, if agents have reached the end of their journey then they stay at desination node
    max_path_length = max([len(path) for path in paths])
    i = 0
    while i < max_path_length and curr_time < T:
        curr_time += 1
        i += 1
        # arm_set will be the set of arms that are visited at the current time
        arm_set = set()
        for agent in agents:
            # Add current vertex to arm_set
            arm_set.add(agent.current_node['arm'])

            # If we have more path left, move to next node
            if i < len(paths[agent.id]):
                agent.move(G.nodes[paths[agent.id][i]])

        rew_per_turn.append(0)
        for arm in arm_set:
            rew_per_turn[-1] += arm.pull(curr_time)

    # We update arm_set
    arm_set = set()
    for agent in agents:
        # Add current vertex to arm_set
        arm_set.add(agent.current_node['arm'])

    # We sample until num_pulls of arm_with_min_pulls doubles
    assert(G.nodes[arm_with_min_pulls]['arm'] in arm_set)
    while G.nodes[arm_with_min_pulls]['arm'].num_pulls < 2 * min_pulls and curr_time < T:
        curr_time += 1
        rew_per_turn.append(0)
        for arm in arm_set:
            rew_per_turn[-1] += arm.pull(curr_time)

    return curr_time, rew_per_turn


"""
---------------------------------------------------------------------------------------------------------------------------------------
    Executable Code
--------------------------------------------------------------------------------------------------------------------------------------- 
"""
seed = int(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
# Note that variations in the results come from an unstable maximum weight matching algorithm in the 'episode' function

# Problem Parameters
T = 100000
K = 1000
M = 20
p = 0.05

# Initialize connected graph
G = nx.erdos_renyi_graph(K, p, seed = seed)
tries = 0
while not nx.is_connected(G) and tries < 10:
    G = nx.erdos_renyi_graph(K, p, seed = seed + tries)
    tries += 1
assert(nx.is_connected(G))

# Assign each vertex an associated arm
for i in G:
    G.nodes[i]['arm'] = Arm(i)
    G.nodes[i]['id']  = i
    G.nodes[i]['prev_node'] = G.nodes[i]

# Initialize agents and assign vertex
agents   = [Agent(i, G.nodes[random.randint(0, K-1)]) for i in range(M)]

# Initialize output file and results directory
if not os.path.exists("seed_" + str(seed)):
    os.makedirs("seed_" + str(seed))

f = open("seed_" + str(seed) + "/{}.txt".format(sys.argv[2]), "w")

# Begin Algorithm
# After the return from each function call, 
# curr_time is the last time step that was run
# And each agent has yet to sample from their current vertex
curr_time = 0
curr_ep   = 0
curr_time, reward_per_turn = initialize()

while curr_time < T:
    curr_ep   += 1
    print("Episode {}".format(curr_ep))
    # f.write("Starting Episode {}, Current time is {}\n".format(curr_ep, curr_time))
    curr_time, new_rewards = episode(curr_time)
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
max_per_turn = sum(sorted([G.nodes[node]['arm'].true_mean for node in G.nodes()], reverse = True)[:M])

print('Total Time: ' + str(curr_time))
f.write('\n----------------------------------------------------\n')
f.write('Total Time: ' + str(curr_time))
f.write("\nNet Reward: " + str(net_reward))
f.write("\nTheoretical Expected Max: " + str(T * max_per_turn))
f.write("\n")
f.close()


# # Plot cumulative reward
# assert(len(reward_per_turn) == curr_time == T)
# plt.plot(range(T), np.cumsum(reward_per_turn), label = 'Observed')
# plt.plot(range(T), [max_per_turn * i for i in range(1, T+1)], label = 'Theoretical Max')
# plt.xlabel("Time")
# plt.ylabel("Cumulative Reward")
# plt.title("Cumulative reward as a function of time")
# plt.legend()
# plt.savefig("seed_" + str(seed) + "/cumulative_reward.png")

# # Plot Cumulative Regret
# plt.clf()
# plt.plot(range(T), np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)))
# plt.xlabel("Time")
# plt.ylabel("Cumulative Regret")
# plt.title("Cumulative regret as a function of time")
# plt.savefig("seed_" + str(seed) + "/cumulative_regret.png")

# # Plot Average Regret
# plt.clf()
# plt.plot(range(T), np.divide(np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)), range(1, T+1)))
# plt.xlabel("Time")
# plt.ylabel("Average Regret")
# plt.title("Average regret as a function of time")
# plt.savefig("seed_" + str(seed) + "/av_regret.png")

# # Visualize graph
# plt.clf()
# nx.draw(G, with_labels = True)
# plt.savefig("seed_" + str(seed) + "/state_graph.png")
