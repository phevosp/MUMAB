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
        self.type = type  # algorithm type {robust or simple}
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
            rew_this_turn += (
                arm.interaction.function(arm_dict[arm]) * true_single_reward
            )
        return rew_this_turn

    def all_agents_sampled(self, agents):
        sampled = True
        for agent in agents:
            sampled = sampled and agent.sampled_episode_req()
        return sampled

    def _episode(self, agents, curr_time):
        """
        Runs one episode of the algorithm.
        Optimal distribution is computed using UCB estimates of each arm.
        Agents move to their assigned destination node, and sample that node until the baseline_arm has its samples doubled.
        """
        if curr_time > 0:
            # Have agents define packages
            for agent in agents:
                agent.define_package()

            # Update UCB values from previous episode/initialization
            for node in self.G:
                if self.type == "simple":
                    self.G.nodes[node]["arm"].update_attributes_simple(
                        agents, curr_time
                    )
                elif self.type == "robust":
                    self.G.nodes[node]["arm"].update_attributes_robust(
                        agents, curr_time
                    )

            # Reset packages
            for agent in agents:
                agent.reset_package()

        # Keep track of reward per turn
        rew_per_turn = []

        # Compute optimal distribution
        distribution, _ = optimal_distribution(
            [self.G.nodes[node]["arm"] for node in self.G], self.params
        )

        # Create list of sampled nodes
        # If a node is to be sampled n times then it appears n times in the list
        # The number of times the node is sampled is of course x_node = x_self.G.nodes[node]['arm'].id
        sampled_nodes = []
        for node in self.G:
            for times in range(
                round(distribution[f"x_{self.G.nodes[node]['arm'].id}"])
            ):
                sampled_nodes.append(node)

        # Note number of pulls of baseline
        sorted_by_pulls = sorted(
            sampled_nodes, key=lambda x: self.G.nodes[x]["arm"].num_pulls
        )
        baseline_arm = sorted_by_pulls[0]
        baseline_pulls = self.G.nodes[baseline_arm]["arm"].num_pulls

        # Note maximum ucb value for edge
        max_ucb = max([self.G.nodes[node]["arm"].ucb for node in sampled_nodes])

        # Initialize edge weights given UCB estimate of each arm.
        # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
        G_directed = nx.DiGraph(self.G)
        for u, v in self.G.edges():
            # Floating point errors incured so flooring at 0
            G_directed.edges[u, v]["weight"] = max(
                max_ucb - self.G.nodes[v]["arm"].ucb, 0
            )
            G_directed.edges[v, u]["weight"] = max(
                max_ucb - self.G.nodes[u]["arm"].ucb, 0
            )

        # For each agent and optimal arm pair compute shortest path to create weights for bipartite graph
        # sp_dict is indexed by (agent_id, node_i) and stores a tuple (path length, actual path)
        # where path is the shortest path between the current node of the agent and the destination node
        sp_dict = {}
        for agent in agents:
            # Compute single source shortest path to all other nodes
            try:
                shortest_path = nx.shortest_path(
                    G_directed, source=agent.current_node["id"], weight="weight"
                )
            except:
                for u, v in G_directed.edges():
                    if G_directed.edges[u, v]["weight"] < 0:
                        print(G_directed.edges[u, v]["weight"])
                assert False

            # Compute single source shortest path length to all other nodes
            shortest_path_length = nx.shortest_path_length(
                G_directed, source=agent.current_node["id"], weight="weight"
            )
            # And then add path to shortest path dictionary for all destination nodes
            for i, dest_node in enumerate(sampled_nodes):
                sp_dict[(agent.id, f"{dest_node}_{i}")] = (
                    shortest_path_length[dest_node],
                    shortest_path[dest_node],
                )

        # Create bipartite graph
        B = nx.Graph()
        B.add_nodes_from([("agent", agent.id) for agent in agents])
        B.add_nodes_from([(f"node_{i}", node) for i, node in enumerate(sampled_nodes)])
        for agent_id, dest_node_str in sp_dict:
            dest_node = int(dest_node_str.split("_")[0])
            index = int(dest_node_str.split("_")[1])
            B.add_edge(
                ("agent", agent_id),
                (f"node_{index}", dest_node),
                weight=sp_dict[(agent_id, dest_node_str)][0],
            )
        assignments = bipartite.minimum_weight_full_matching(
            B, top_nodes=[("agent", agent.id) for agent in agents], weight="weight"
        )

        # Create list paths where paths[i] is the path for agent i
        paths = [[] for _ in agents]

        baseline_agent = None

        for agent in agents:
            (node_name, dest_node) = assignments[("agent", agent.id)]
            index = int(node_name.split("_")[1])
            paths[agent.id] = sp_dict[(agent.id, f"{dest_node}_{index}")][1]
            agent.set_target_path(paths[agent.id])
            if dest_node == baseline_arm:
                if (
                    not baseline_agent
                    or agent.get_path_len() < baseline_agent.get_path_len()
                ):
                    baseline_agent = agent

        theoretical_max_episode = baseline_agent.get_path_len() + baseline_pulls - 1
        for agent in agents:
            agent.set_sample_req(baseline_pulls)
        

        # determines transition time interval, starts at the current time and goes until the minimum of self.T or curr_time + max_path_length
        trans_t = [curr_time, 0]

        all_agents_reached = False

        episode_len_bound = (
            theoretical_max_episode * (self.params.alpha + 1)
            if self.type == "robust"
            else theoretical_max_episode
        )
            
        episode_not_over = (curr_time - trans_t[0]) < episode_len_bound
        if self.type == "simple":            
            episode_not_over = not self.all_agents_sampled(agents)
        
        while (
            not all_agents_reached
            and curr_time < self.T
            and episode_not_over
        ):
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
                if agent.current_node["arm"] not in arm_dict:
                    arm_dict[agent.current_node["arm"]] = 1
                    arm_dict_agents[agent.current_node["arm"]] = [agent]
                else:
                    arm_dict[agent.current_node["arm"]] += 1
                    arm_dict_agents[agent.current_node["arm"]].append(agent)

            # arm_dict is number of agents on each arm
            # arm_dict_agents is the agents at each arm
            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))
        
        
            episode_not_over = (curr_time - trans_t[0]) < episode_len_bound
            if self.type == "simple":            
                episode_not_over = not self.all_agents_sampled(agents)


        # Update end of transition interval
        trans_t[1] = min(curr_time, self.T - 1)
        # We update arm_dict
        arm_dict = {}

        # the agents at the arm
        arm_dict_agents = {}
        for agent in agents:
            # Add current vertex to arm_dict
            if agent.current_node["arm"] not in arm_dict:
                arm_dict[agent.current_node["arm"]] = 1
                arm_dict_agents[agent.current_node["arm"]] = [agent]
            else:
                arm_dict[agent.current_node["arm"]] += 1
                arm_dict_agents[agent.current_node["arm"]].append(agent)

        # We sample until num_pulls of baseline_arm doubles
        if self.G.nodes[baseline_arm]["arm"] not in arm_dict:
            assert curr_time == self.T or (curr_time - trans_t[0]) == episode_len_bound

        episode_not_over = (curr_time - trans_t[0]) < episode_len_bound
        if self.type == "simple":            
            episode_not_over = not self.all_agents_sampled(agents)

        while episode_not_over and curr_time < self.T:
            curr_time += 1
            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))

            episode_not_over = (curr_time - trans_t[0]) < episode_len_bound
            if self.type == "simple":            
                episode_not_over = not self.all_agents_sampled(agents)

        # for tracking agent movement over time
        allocation = []
        for agent in agents:
            allocation.append(agent.current_node["arm"].id)

        return curr_time, rew_per_turn, trans_t, allocation

    def run(self, max_reward_per_turn):
        # Reset arms
        for i in self.G:
            self.G.nodes[i]["arm"].reset()

        # Initialize agents and assign vertex
        agents = [
            Agent(
                i,
                self.G.nodes[random.randint(0, self.K - 1)],
                self.G,
                self.params.agent_std_dev[i],
                self.params.agent_bias[i],
                self.params.agent_move_prob[i],
                self.params.agent_sample_prob[i],
                self.params.agent_move_alpha[i],
                self.params.agent_move_beta[i],
                self.params.agent_sample_alpha[i],
                self.params.agent_sample_beta[i]
            )
            for i in range(self.M)
        ]

        # Begin Algorithm
        # After the return from each function call,
        # curr_time is the last time step that was run
        # And each agent has yet to sample from their current vertex
        curr_time = 0
        curr_ep = 0

        # INITIALIZE ARMS WITH ONE SAMPLE
        for arm in self.G:
            self.G.nodes[arm]["arm"].update_attributes_hack(len(agents), self.type)
        reward_per_turn = []

        # List of transition intervals
        transition_intervals = []

        episode_allocations = []
        with tqdm(total=self.T) as pbar:
            while curr_time < self.T:
                curr_ep += 1
                curr_time, new_rewards, trans_t, allocation = self._episode(
                    agents, curr_time
                )
                reward_per_turn += new_rewards
                transition_intervals.append(trans_t)
                episode_allocations.append(allocation)
                pbar.update(curr_time - pbar.n)
        pbar.close()

        # Get results
        regret = max_reward_per_turn - np.array(reward_per_turn)
        return regret, transition_intervals



class MAB_INDV:
    def __init__(self, G, params):
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
            rew_this_turn += (
                arm.interaction.function(arm_dict[arm]) * true_single_reward
            )
        return rew_this_turn


    def plan_on_agent(self, agent, curr_time):
        if curr_time > 0:
            for node in self.G:
                self.G.nodes[node]["arm"].update_attributes(agent, curr_time)
        
        ucb_ranking = sorted(
            self.G.nodes,
            key=lambda x: self.G.nodes[x]["arm"].Arms[agent.id].ucb,
            reverse=True,
        )
        optimal_arm = ucb_ranking[0]
        
        G_directed = self.get_G_directed(agent.id, optimal_arm)
        
        try:
            shortest_path = nx.shortest_path(
                G_directed, source=agent.current_node["id"], weight="weight"
            )
        except:
            for u, v in G_directed.edges():
                if G_directed.edges[u, v]["weight"] < 0:
                    print(G_directed.edges[u, v]["weight"])
            assert False
            
        agent.set_target_path(shortest_path[optimal_arm])        
        agent.set_sample_req(self.G.nodes[optimal_arm]['arm'].Arms[agent.id].num_samples)
        

    def get_G_directed(self, agent_id, selected_arm):
        # Note maximum ucb value for edge
        max_ucb = self.G.nodes[selected_arm]["arm"].Arms[agent_id].ucb

        # Initialize edge weights given UCB estimate of each arm.
        # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
        G_directed = nx.DiGraph(self.G)
        for u, v in self.G.edges():
            # Floating point errors incured so flooring at 0
            G_directed.edges[u, v]["weight"] = max(
                max_ucb - self.G.nodes[v]["arm"].Arms[agent_id].ucb, 0
            )
            G_directed.edges[v, u]["weight"] = max(
                max_ucb - self.G.nodes[u]["arm"].Arms[agent_id].ucb, 0
            )
        return G_directed
    
    def _time_step(self, agents, curr_time):
        reward = 0
        locations = {}
        for agent in agents:
            if not agent.at_target_pose():
                agent.move()
            else:
                if agent.sampled_episode_req():
                    self.plan_on_agent(agent, curr_time)                    
                    agent.move()
            if agent.current_node["id"] in locations:
                locations[agent.current_node["id"]].append(agent)
            else:
                locations[agent.current_node["id"]] = [agent]

        total_reward = 0
        for loc in locations:
            arm = self.G.nodes[loc]["arm"].Arms[locations[loc][0].id]                
            reward = arm.pull(len(locations[loc]))
            for agent in locations[loc]:
                agent.sample(reward)
            
            total_reward += (
                arm.interaction.function(len(locations[loc])) * reward
            )

        return total_reward
                
                


    def run(self, max_reward_per_turn):
        # Reset arms
        for i in self.G:
            self.G.nodes[i]["arm"].reset()

        # Initialize agents and assign vertex
        agents = [
            Agent(
                i,
                self.G.nodes[random.randint(0, self.K - 1)],
                self.G,
                self.params.agent_std_dev[i],
                self.params.agent_bias[i],
                self.params.agent_move_prob[i],
                self.params.agent_sample_prob[i],
                self.params.agent_move_alpha[i],
                self.params.agent_move_beta[i],
                self.params.agent_sample_alpha[i],
                self.params.agent_sample_beta[i]
            )
            for i in range(self.M)
        ]

        # Begin Algorithm
        # After the return from each function call,
        # curr_time is the last time step that was run
        # And each agent has yet to sample from their current vertex
        curr_time = 0
        curr_ep = 0

        # INITIALIZE ARMS WITH ONE SAMPLE
        for arm in self.G:
            self.G.nodes[arm]["arm"].update_attributes_hack()

        regret = []
        for t in tqdm(range(self.T)):
            reward = self._time_step(agents, t)
            regret.append(max_reward_per_turn - reward)
        return regret, None


def getMAB(type, G, G_, params):
    if type == "indv":
        return MAB_INDV(G_, params)

    return MAB(type, G, params)
