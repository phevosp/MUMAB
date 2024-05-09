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
from tqdm import tqdm


from MUMAB.objects.Agent import Agent
from .utils.OptimalDistribution import optimal_distribution

class MAB:
    def __init__(self, type, G, params):
        self.type = type        #algorithm type {original [min], median, max}
        self.params = params
        self.G = G
        self.T = params.T
        self.K = params.K
        self.M = params.M


    def _step(self, arm_dict, arm_dict_agents, curr_time):
        rew_this_turn = 0
        for arm in arm_dict:
            # Pull the arm to get the true reward at time curr_time
            true_single_reward = arm.pull(arm_dict[arm])

            for agent in arm_dict_agents[arm]:
                # Observe reward
                agent.sample(true_single_reward)    

            # Add the theoretical reward per turn, assuming all agents sampled for fair comparison
            rew_this_turn += arm.interaction.function(arm_dict[arm]) * true_single_reward
        return rew_this_turn

    def _initialize(self, agents):
        """
        Initializes the approximations for each vertex by having each agent run DFS until all the vertices are visited at least once.
        """
        curr_time = 0
        # Keep track of reward per turn
        rew_per_turn = []

        # Initialize list of visited nodes
        visited    = [False for node in self.G]
        for agent in agents:
            visited[agent.current_node['id']] = True

        while not all(visited):
            curr_time += 1
            # arm_dict will be the collection of arms that are visited at the current time
            arm_dict = {}

            # the agents at the arm
            arm_dict_agents = {}

            for agent in agents:
                # Add current vertex to arm_dict
                if agent.current_node['arm'] not in arm_dict:
                    arm_dict[agent.current_node['arm']] = 1
                    arm_dict_agents[agent.current_node['arm']] = [agent]
                else:
                    arm_dict[agent.current_node['arm']] += 1
                    arm_dict_agents[agent.current_node['arm']].append(agent)

                # Move to first neighbor that has not been visited
                moved = False
                for neighbor_id in nx.all_neighbors(self.G, agent.current_node['id']):
                    if not visited[neighbor_id]:
                        self.G.nodes[neighbor_id]['prev_node'] = agent.current_node

                        # target path is just next node 
                        agent.set_target_path([neighbor_id])

                        #try to move to next node
                        agent.move()

                        # Mark current node as visited and update moved to be true given we tried to move
                        visited[agent.current_node['id']] = True
                        moved                             = True
                        break
                
                # If we didn't move forward then move backwards
                if not moved:    
                    agent.set_target_path([agent.current_node['prev_node']['id']])
                    agent.move()

            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))

        # Sample current arms as well
        curr_time += 1
        for agent in agents:
            # Add current vertex to arm_dict
            if agent.current_node['arm'] not in arm_dict:
                arm_dict[agent.current_node['arm']] = 1
                arm_dict_agents[agent.current_node['arm']] = [agent]
            else:
                arm_dict[agent.current_node['arm']] += 1
                arm_dict_agents[agent.current_node['arm']].append(agent)


        rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))

        return curr_time, rew_per_turn 
    
    def _episode(self, agents, curr_time):
        """
            Runs one episode of the algorithm. 
            Optimal distribution is computed using UCB estimates of each arm.
            Agents move to their assigned destination node, and sample that node until the baseline_arm has its samples doubled.
            If type == 'original': baseline_arm is the arm with the least number of pulls
            If type == 'median':   baseline_arm is the arm with the median number of pulls
            If type == 'max':      baseline_arm is the arm with the most number of pulls
        """
        # Have agents define packages
        for agent in agents:
            agent.define_package()

        # Update UCB values from previous episode/initialization
        for node in self.G:
            self.G.nodes[node]['arm'].update_attributes(agents, curr_time)

        # Reset packages
        for agent in agents:
            agent.reset_package()

        # Keep track of reward per turn
        rew_per_turn = []

        # Compute optimal distribution
        distribution, _ = optimal_distribution([self.G.nodes[node]['arm'] for node in self.G], self.params)            

        # Create list of sampled nodes
        # If a node is to be sampled n times then it appears n times in the list
        # The number of times the node is sampled is of course x_node = x_self.G.nodes[node]['arm'].id
        sampled_nodes = []
        for node in self.G:
            for times in range(round(distribution[f"x_{self.G.nodes[node]['arm'].id}"])):
                sampled_nodes.append(node)

        # Note number of pulls of baseline
        sorted_by_pulls = sorted(sampled_nodes, key = lambda x : self.G.nodes[x]['arm'].num_pulls)
        if self.type == 'original':
            baseline_arm = sorted_by_pulls[0]
        elif self.type == 'median':
            baseline_arm = sorted_by_pulls[len(sampled_nodes) // 2]
        elif self.type == 'max':
            baseline_arm = sorted_by_pulls[-1]

        baseline_pulls = self.G.nodes[baseline_arm]['arm'].num_pulls
        # f.write("Baseline Arm: {}\n".format(baseline_arm))

        # Note maximum ucb value for edge
        max_ucb = max([self.G.nodes[node]['arm'].ucb for node in sampled_nodes])
        
        # Initialize edge weights given UCB estimate of each arm.
        # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
        G_directed = nx.DiGraph(self.G)
        for (u, v) in self.G.edges():
            # Floating point errors incured so flooring at 0
            G_directed.edges[u, v]["weight"] = max(max_ucb - self.G.nodes[v]['arm'].ucb, 0)
            G_directed.edges[v, u]["weight"] = max(max_ucb - self.G.nodes[u]['arm'].ucb, 0)

        # For each agent and optimal arm pair compute shortest path to create weights for bipartite graph
        # sp_dict is indexed by (agent_id, node_i) and stores a tuple (path length, actual path)
        # where path is the shortest path between the current node of the agent and the destination node
        sp_dict = {}
        for agent in agents:
            # Compute single source shortest path to all other nodes
            try:
                shortest_path        = nx.shortest_path(G_directed, source = agent.current_node['id'], weight = "weight") 
            except:
                for (u, v) in G_directed.edges():
                    if G_directed.edges[u, v]["weight"] < 0:
                        print(G_directed.edges[u, v]["weight"])
                assert(False)

            # Compute single source shortest path length to all other nodes
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

        baseline_agent = None

        for agent in agents:
            (node_name, dest_node) = assignments[('agent', agent.id)]
            index  = int(node_name.split('_')[1])
            paths[agent.id] = sp_dict[(agent.id, f"{dest_node}_{index}")][1]
            agent.set_target_path(paths[agent.id])
            if dest_node == baseline_arm:
                if not baseline_agent or agent.get_path_len() < baseline_agent.get_path_len():
                    baseline_agent = agent                

        theoretical_max_episode = baseline_agent.get_path_len() + 2*baseline_pulls - 1
        # f.write("Paths: {}\n".format(paths))

        # determines transition time interval, starts at the current time and goes until the minimum of self.T or curr_time + max_path_length
        trans_t = [curr_time, 0]
        
        all_agents_reached = False

        episode_len_bound = theoretical_max_episode*(self.params.alpha + 1)

        while not all_agents_reached and curr_time < self.T and (curr_time - trans_t[0]) < episode_len_bound:
            curr_time += 1
            # arm_dict will be the set of arms that are visited at the current time
            arm_dict = {}

            # the agents at the arm
            arm_dict_agents = {}

            all_agents_reached = True
            for agent in agents:
                # If we have more path left, move to next node
                agent.move()
                if not agent.at_target_pose():
                    all_agents_reached = False

                # Then add current vertex to arm_dict
                if agent.current_node['arm'] not in arm_dict:
                    arm_dict[agent.current_node['arm']] = 1
                    arm_dict_agents[agent.current_node['arm']] = [agent]
                else:
                    arm_dict[agent.current_node['arm']] += 1
                    arm_dict_agents[agent.current_node['arm']].append(agent)
        

            # arm_dict is number of agents on each arm
            # arm_dict_agents is the agents at each arm
            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))
    
        # Update end of transition interval
        trans_t[1] = min(curr_time, self.T - 1)       
        # We update arm_dict
        arm_dict = {}

        # the agents at the arm
        arm_dict_agents = {}
        for agent in agents:
            # Add current vertex to arm_dict
            if agent.current_node['arm'] not in arm_dict:
                arm_dict[agent.current_node['arm']] = 1
                arm_dict_agents[agent.current_node['arm']] = [agent]
            else:
                arm_dict[agent.current_node['arm']] += 1
                arm_dict_agents[agent.current_node['arm']].append(agent)

        # We sample until num_pulls of baseline_arm doubles
        if self.G.nodes[baseline_arm]['arm'] not in arm_dict:
            assert(curr_time == self.T or (curr_time - trans_t[0]) == episode_len_bound)

        while (curr_time - trans_t[0]) < episode_len_bound and curr_time < self.T:
            curr_time += 1
            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))

        return curr_time, rew_per_turn, trans_t
    
    def run(self, f):
        # Reset arms
        for i in self.G:
            self.G.nodes[i]['arm'].reset()

        # Initialize agents and assign vertex
        agents   = [Agent(i, 
                          self.G.nodes[random.randint(0, self.K-1)], 
                          self.G, self.params.agent_std_dev[i], 
                          self.params.agent_bias[i], 
                          self.params.agent_move_prob[i], 
                          self.params.agent_sample_prob[i], 
                          self.params.agent_move_gamma[i], 
                          self.params.agent_sample_gamma[i]) 
                    for i in range(self.M)]

        # Begin Algorithm
        # After the return from each function call, 
        # curr_time is the last time step that was run
        # And each agent has yet to sample from their current vertex
        curr_time = 0
        curr_ep   = 0
        curr_time, reward_per_turn = self._initialize(agents)


        # List of transition intervals
        transition_intervals = []
        with tqdm(total=self.T) as pbar:
            while curr_time < self.T:
                curr_ep   += 1
                # print("Episode {}".format(curr_ep))
                # f.write("Starting Episode {}, Current time is {}\n".format(curr_ep, curr_time))
                curr_time, new_rewards, trans_t = self._episode(agents, curr_time)
                reward_per_turn += new_rewards
                transition_intervals.append(trans_t)
                pbar.update(curr_time - pbar.n)
        pbar.close()

        # Get results
        for node in self.G.nodes():
            arm = self.G.nodes[node]['arm']
            f.write('\nArm ' + str(arm.id))
            f.write("\nTrue Mean: " + str(arm.true_mean))
            f.write("\nEstimated Mean: " +  str(arm.estimated_mean))
            f.write("\nUCB Value: " + str(arm.ucb))
            f.write("\nNum Pulls: " + str(arm.num_pulls))

        return reward_per_turn, curr_time, transition_intervals

class MAB_indv:
    def __init__(self, G, params):
        self.G = G
        self.T = params.T
        self.K = params.K
        self.M = params.M

    def _step(self, curr_time, agents, arm_dict):
        rew_this_turn = 0
        for arm in arm_dict:
            rew_this_turn += arm.pull_individual(curr_time, agents, arm_dict[arm])
        return rew_this_turn

    def _initialize(self, agents):
        """
            Initializes the approximations for each vertex by having each agent run DFS until it has visited all the vertices.
        """
        curr_time = 1
        # Keep track of reward per turn
        rew_per_turn = []

        # Initialize list of visited nodes for each agent
        visited    = {agent.id:[False for node in self.G] for agent in agents}
        for agent in agents:
            # print(agent.id, agent.current_node['id'])
            visited[agent.id][agent.current_node['id']] = True

        # Initialize list of previous nodes for each agent
        prev_nodes = {agent.id:{} for agent in agents}

        # While there is some agent that has not visited all nodes
        while not all([sum(visited[agent.id]) == len(self.G) for agent in agents]):
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
                for neighbor_id in nx.all_neighbors(self.G, agent.current_node['id']):
                    if not visited[agent.id][neighbor_id]:
                        prev_nodes[agent.id][neighbor_id] = agent.current_node['id']
                        agent.set_target_path([self.G.nodes[neighbor_id]])
                        agent.move()

                        # Immediately mark as visited, though we haven't yet sampled reward
                        visited[agent.id][agent.current_node['id']] = True
                        moved                             = True
                        break
                
                # If we didn't move forward then move backwards
                if not moved:  
                    prev_node_id = prev_nodes[agent.id][agent.current_node['id']]
                    agent.move(self.G.nodes[prev_node_id])

            rew_per_turn.append(self._step(curr_time, agents, arm_dict))
        
        # Sample current arms as well
        for agent in agents:
            if agent.current_node['arm'] not in arm_dict:
                arm_dict[agent.current_node['arm']] = [agent.id]
            else:
                arm_dict[agent.current_node['arm']].append(agent.id)

        rew_per_turn.append(self._step(curr_time, agents, arm_dict))

        # Double check all vertices have been visited
        for agent in agents:
            assert(sorted(agent.estimated_mean_dict.keys()) == sorted(self.G.nodes()))

        return curr_time, rew_per_turn
    
    def _episode(self, agents, curr_time):
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
            G_directed = nx.DiGraph(self.G)
            for (u, v) in self.G.edges():
                G_directed.edges[u, v]["weight"] = max_ucb - agent.ucb_dict[self.G.nodes[v]['arm'].id]
                G_directed.edges[v, u]["weight"] = max_ucb - agent.ucb_dict[self.G.nodes[u]['arm'].id]
        
            # Compute single source shortest path and add to dictionary
            paths[agent.id]      = nx.shortest_path(G_directed, source = agent.current_node['id'], target = opt_arm, weight = "weight")
            agent.set_target_path(paths[agent.id])

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

        # determines transition time interval, starts at the current time and goes until the minimum of self.T or curr_time + max_path_length
        trans_t = (curr_time, 0)

        all_agents_reached = False
        while not all_agents_reached and curr_time < self.T:
            curr_time += 1
            i += 1
            # arm_dict will be the set of arms that are visited at the current time
            arm_dict = {}
            all_agents_reached = True

            for agent in agents:
                # Add current vertex to arm_dict
                if agent.current_node['arm'] not in arm_dict:
                    arm_dict[agent.current_node['arm']] = [agent.id]
                else:
                    arm_dict[agent.current_node['arm']].append(agent.id)

                # If we have more path left, move to next node
                agent.move()
                if not agent.at_target_pose():
                    all_agents_reached = False


            rew_per_turn.append(self._step(curr_time, agents, arm_dict))

        trans_t[1] = curr_time
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
            assert(curr_time == self.T)

        while baseline_agent.num_pulls_dict[baseline_arm] < 2 * baseline_pulls and curr_time < self.T:
            curr_time += 1
            rew_per_turn.append(self._step(curr_time, agents, arm_dict))

        return curr_time, rew_per_turn, trans_t
    
    def run(self, f):
        """
            Runs the individual learner algorithm on the inputted graph G for T time steps.
        """
        # Initialize graph using individual arm objects
        for i in self.G:
            self.G.nodes[i]['id']  = i
            self.G.nodes[i]['prev_node'] = self.G.nodes[i]
        
        # Initialize agents and assign vertex
        agents   = [Agent(i, self.G.nodes[random.randint(0, self.K-1)], self.G) for i in range(self.M)]

        # Begin Algorithm
        # After the return from each function call, 
        # curr_time is the last time step that was run
        # And each agent has yet to sample from their current vertex
        curr_time = 0
        curr_ep   = 0
        curr_time, reward_per_turn = self._initialize(agents)

        # List of transition intervals
        transition_intervals = []

        while curr_time < self.T:
            curr_ep   += 1
            # print("Episode {}".format(curr_ep))
            # f.write("Starting Episode {}, Current time is {}\n".format(curr_ep, curr_time))
            curr_time, new_rewards, trans_t = self._episode(agents, curr_time)
            reward_per_turn += new_rewards
            transition_intervals.append(trans_t)

        # Get results
        for agent in agents:
            f.write('--------------------------------------------------')
            f.write('\nAgent ' + str(agent.id))
            for node in self.G.nodes():
                arm = self.G.nodes[node]['arm']
                f.write('\nArm ' + str(arm.id))
                f.write("\nTrue Mean: " + str(arm.true_mean))
                f.write("\nEstimated Mean: " +  str(agent.estimated_mean_dict[arm.id]))
                f.write("\nUCB Value: " + str(agent.ucb_dict[arm.id]))
                f.write("\nNum Pulls: " + str(agent.num_pulls_dict[arm.id]))

        return reward_per_turn, curr_time, transition_intervals

def getMAB(type, G, params):
    if type == 'indv':
        return MAB_indv(G, params)
    
    return MAB(type, G, params)