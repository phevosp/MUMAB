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
import math


from MUMAB.objects.Agent import Agent
from .utils.OptimalDistribution import optimal_distribution


class MAB:
    def __init__(self, type, G, params):
        self.type = type  # algorithm type {simple, robust, max, or UCRL2}
        self.params = params
        self.G = G
        self.T = params.T
        self.K = params.K
        self.M = params.M

    def _step(self, arm_dict, arm_dict_agents, curr_time):
        rew_this_turn = 0
        for arm in arm_dict:

            for agent in arm_dict_agents[arm]:
                agent.sample(arm.pull(arm_dict[arm]))

            # Add the theoretical reward per turn, assuming all agents sampled for fair comparison
            rew_this_turn += arm.interaction.function(arm_dict[arm]) * arm.true_mean
        return rew_this_turn

    def _evi_value_iteration(self, destination_map, epsilon):
        """
        epsilon: The stopping condition.
        """
        assert epsilon > 0
        u = np.zeros(self.K)
        policy = {}

        iter_count = 0

        # max_iter: Hard upper-bound of the number of iterations, to ensure the algorithm does not fall into infinite loop
        max_iter = np.min([1e4, 1 / epsilon])

        while iter_count <= max_iter:
            iter_count += 1

            u_old = np.array(u)

            for node in self.G.nodes:
                # Default to staying
                best_nb_u = u_old[node]
                policy[node] = node
                # Check if better to move to a neighbor
                for neighb in self.G.neighbors(node):
                    if u_old[neighb] >= best_nb_u:
                        best_nb_u = u_old[neighb]
                        policy[node] = neighb

                arm = self.G.nodes[node]["arm"]
                u[node] = (
                    arm.interaction.function(destination_map[node] + 1) * arm.ucb
                    - arm.interaction.function(destination_map[node]) * arm.ucb
                ) + best_nb_u
                # print(iter_count, u)

            if np.max(u - u_old) - np.min(u - u_old) < epsilon:
                # print('Gap',np.max(u-u_old) - np.min(u-u_old))
                break

        return policy, u

    def _episode_UCRL2(self, agents, curr_time):
        """
        Runs one episode of the algorithm using the modified UCRL2 algorithm
        """
        if curr_time > 0:
            # Have agents define packages
            for agent in agents:
                agent.define_package()

            # Update UCB values from previous episode/initialization
            for node in self.G:
                self.G.nodes[node]["arm"].update_attributes_UCRL2(
                    agents, curr_time, self.K, len(self.G.edges), self.params.delta
                )

            # Reset packages
            for agent in agents:
                agent.reset_package()

        # Keep track of reward per turn
        rew_per_turn = []

        # Now, for each agent perform policy iteration.
        epsilon = 1 / curr_time if curr_time > 0 else 1
        destination_map = {
            node: 0 for node in self.G.nodes
        }  # destination map takes into account decision of previous agents
        for agent in agents:
            policy, _ = self._evi_value_iteration(destination_map, epsilon)
            agent.set_policy(policy)
            for node in policy:
                if policy[node] == node:
                    destination_map[node] += 1

        baseline_pulls = [self.G.nodes[node]["arm"].num_pulls for node in self.G.nodes]
        visits_this_episode = [0 for node in self.G.nodes]
        episode_not_over = True
        while curr_time < self.T and episode_not_over:
            curr_time += 1
            arm_dict = {}
            arm_dict_agents = {}
            for agent in agents:
                # Move to next node via policy
                agent.move_via_policy()

                # Then add current vertex to arm_dict
                if agent.current_node["arm"] not in arm_dict:
                    arm_dict[agent.current_node["arm"]] = 1
                    arm_dict_agents[agent.current_node["arm"]] = [agent]
                else:
                    arm_dict[agent.current_node["arm"]] += 1
                    arm_dict_agents[agent.current_node["arm"]].append(agent)

                # Update visits_this_episode
                visits_this_episode[agent.current_node["id"]] += 1
                # Also update episode_not_over if arm pull counts exceeds double
                if (
                    visits_this_episode[agent.current_node["id"]]
                    >= baseline_pulls[agent.current_node["id"]]
                ):
                    episode_not_over = False

            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))

        # for tracking agent movement over time
        allocation = []
        for agent in agents:
            allocation.append(agent.current_node["arm"].id)

        return curr_time, rew_per_turn, 0, allocation

    def _episode_pulls_req_met(self, sampled_nodes):
        """
        Returns True if all arms have satisfied their pull requirement
        """
        for node in sampled_nodes:
            if not self.G.nodes[node]["arm"].episode_pulls_req_met():
                return False
        return True

    def _episode(self, agents, curr_time):
        """
        Runs one episode of the algorithm.
        Optimal distribution is computed using UCB estimates of each arm.
        """
        if curr_time > 0:
            # Have agents define packages
            for agent in agents:
                agent.define_package()

            # Update UCB values from previous episode/initialization
            for node in self.G:
                self.G.nodes[node]["arm"].update_attributes(agents, curr_time)

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
        sampled_nodes = []
        for node in self.G:
            for times in range(
                round(distribution[f"x_{self.G.nodes[node]['arm'].id}"])
            ):
                sampled_nodes.append(node)

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

        # Set paths for each agent
        paths = [[] for _ in agents]
        for agent in agents:
            (node_name, dest_node) = assignments[("agent", agent.id)]
            index = int(node_name.split("_")[1])
            paths[agent.id] = sp_dict[(agent.id, f"{dest_node}_{index}")][1]
            agent.set_target_path(paths[agent.id])

        # Records the length of the transition phase
        trans_t = [curr_time, 0]

        # Define stopping criteria
        sorted_by_pulls = sorted(
            sampled_nodes, key=lambda x: self.G.nodes[x]["arm"].num_pulls
        )
        baseline_arm = sorted_by_pulls[-1] if self.type == "max" else sorted_by_pulls[0]
        baseline_pulls = self.G.nodes[baseline_arm]["arm"].num_pulls
        for node in self.G:
            self.G.nodes[node]["arm"].set_episode_pulls_req(baseline_pulls)

        all_agents_reached = False
        episode_not_over = not self._episode_pulls_req_met(sampled_nodes)

        # Execute transition
        while not all_agents_reached and curr_time < self.T and episode_not_over:
            curr_time += 1
            arm_dict = {}
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

            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))
            episode_not_over = not self._episode_pulls_req_met(sampled_nodes)

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
            assert curr_time == self.T

        episode_not_over = not self._episode_pulls_req_met(sampled_nodes)

        while episode_not_over and curr_time < self.T:
            curr_time += 1
            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))
            episode_not_over = not self._episode_pulls_req_met(sampled_nodes)

        # for tracking agent movement over time
        allocation = []
        for agent in agents:
            allocation.append(agent.current_node["arm"].id)

        return curr_time, rew_per_turn, trans_t, allocation

    def _initialization(self, agents, num_samples):
        """
        Runs an initialization phase to obtain num_samples samples for each arm
        """
        curr_time = 0
        rew_per_turn = []

        # Set arm pull requirements
        for node in self.G:
            self.G.nodes[node]["arm"].set_episode_pulls_req(num_samples)

        # Track unvisited arms
        current_nodes = [agent.current_node["id"] for agent in agents]
        unvisited_arms = set(self.G.nodes) - set(current_nodes)

        # Define stopping condition
        initialization_not_over = not self._episode_pulls_req_met(self.G.nodes)

        # Each agent selects an unvisited arm and traverses to it and back
        while initialization_not_over:
            curr_time += 1
            arm_dict = {}
            arm_dict_agents = {}
            for agent in agents:
                if not agent.at_target_pose():
                    agent.move()
                else:
                    # If no more arms to visit, set a new destination and move
                    if (
                        agent.get_current_node()["arm"].episode_pulls_req_met()
                        and len(unvisited_arms) != 0
                    ):
                        destination_node = unvisited_arms.pop()
                        # Calculate shortest path to destination
                        path = nx.shortest_path(
                            self.G,
                            source=agent.current_node["id"],
                            target=destination_node,
                        )
                        agent.set_target_path(path)
                        agent.move()

                # Add current node to list
                if agent.current_node["arm"] not in arm_dict:
                    arm_dict[agent.current_node["arm"]] = 1
                    arm_dict_agents[agent.current_node["arm"]] = [agent]
                else:
                    arm_dict[agent.current_node["arm"]] += 1
                    arm_dict_agents[agent.current_node["arm"]].append(agent)

            # Update rewards
            rew_per_turn.append(self._step(arm_dict, arm_dict_agents, curr_time))
            # And check if initialization is over
            initialization_not_over = not self._episode_pulls_req_met(self.G.nodes)

        return curr_time, rew_per_turn

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
                self.params.agent_sample_beta[i],
            )
            for i in range(self.M)
        ]

        min_agent_sample_prob = min(self.params.agent_sample_prob)
        robust_initialization_samples = 4 * math.log(self.T) / min_agent_sample_prob**2
        with tqdm(total=self.T) as pbar:
            # Initialize arms
            initialization_samples = (
                robust_initialization_samples if self.type == "robust" else 1
            )
            curr_time, reward_per_turn = self._initialization(
                agents, initialization_samples
            )
            curr_ep = 0
            transition_intervals = []
            episode_allocations = []
            while curr_time < self.T:
                curr_ep += 1
                curr_time, new_rewards, trans_t, allocation = (
                    self._episode(agents, curr_time)
                    if self.type != "UCRL2"
                    else self._episode_UCRL2(agents, curr_time)
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
            for agent in arm_dict_agents[arm]:
                # Observe reward
                reward = arm.pull(arm_dict[arm])
                agent.sample(reward)

            # Add the theoretical reward per turn, assuming all agents sampled for fair comparison
            rew_this_turn += arm.interaction.function(arm_dict[arm]) * arm.true_mean
        return rew_this_turn

    def plan_on_agent(self, agent, curr_time):

        # Update UCB values
        agent.define_package()
        for node in set(agent.arm_list):
            self.G.nodes[node]["arm"].update_attributes(agent, curr_time)
        agent.reset_package()

        try:
            # If still in initialization
            target_arm = agent.to_initialize().pop()
            target_pulls = 1
        except:
            # Otherwise, episode
            ucb_ranking = sorted(
                self.G.nodes,
                key=lambda x: self.G.nodes[x]["arm"].Arms[agent.id].ucb,
                reverse=True,
            )
            target_arm = ucb_ranking[0]
            target_pulls = self.G.nodes[target_arm]["arm"].Arms[agent.id].num_pulls

        G_directed = self.get_G_directed(agent.id, target_arm)
        try:
            shortest_path = nx.shortest_path(
                G_directed, source=agent.current_node["id"], weight="weight"
            )
        except:
            for u, v in G_directed.edges():
                if G_directed.edges[u, v]["weight"] < 0:
                    print(G_directed.edges[u, v]["weight"])
            assert False

        agent.set_target_path(shortest_path[target_arm])
        agent.set_episode_pulls_req(target_pulls)

    def get_G_directed(self, agent_id, selected_arm):
        # Note maximum ucb value for edge
        max_ucb = self.G.nodes[selected_arm]["arm"].Arms[agent_id].ucb

        # Initialize edge weights given UCB estimate of each arm.
        # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
        G_directed = nx.DiGraph(self.G)
        for u, v in self.G.edges():
            # Floating point errors incurred so flooring at 0
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
                if agent.episode_pulls_req_met():
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
                arm.interaction.function(len(locations[loc])) * arm.true_mean
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
                self.params.agent_sample_beta[i],
            )
            for i in range(self.M)
        ]

        # Begin Algorithm
        # After the return from each function call,
        # curr_time is the last time step that was run
        # And each agent has yet to sample from their current vertex
        curr_time = 0
        curr_ep = 0

        regret = []
        for t in tqdm(range(self.T)):
            reward = self._time_step(agents, t)
            regret.append(max_reward_per_turn - reward)
        return regret, None


def getMAB(type, G, G_, params):
    if type == "indv":
        return MAB_INDV(G_, params)

    return MAB(type, G, params)
