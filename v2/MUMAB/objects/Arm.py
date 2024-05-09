import numpy as np
import random
from MUMAB.objects.MultiAgentInteraction import MultiAgentInteractionInterface


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
    def __init__(self, id, interaction):
        self.id             :int   = id
        self.true_mean      :float = random.random() * 0.75 + 0.25
        self.num_pulls      :int   = 0                               # Number of pulls, to be used when calculating confidence radius
        self.num_samples    :int   = 0                               # Number of samples, to be used when calculating mean reward               
        self.total_reward   :int   = 0
        self.estimated_mean :int   = 0
        self.conf_radius    :int   = 0
        self.ucb            :int   = 0
        self.interaction    : MultiAgentInteractionInterface = interaction

    def get_reward(self):
        return np.clip(np.random.normal(loc = self.true_mean, scale = 0.1), 0, 2)
    
    def pull(self, num_agents):
        single_reward = self.get_reward()
        return single_reward

    def update_attributes(self, agents, time):
        total_episode_reward = 0     # Total episode reward
        total_episode_counts = 0     # Total episode sampling counts

        sampling_intervals   = []    # Intervals of when agents were sampling the arm
        earliest_obs         = time  # Earliest time when an agent sampled
        latest_obs           = 0     # Latest time when an agent sampled
        total_episode_pulls  = 0     # Total number of pulls in the episode

        for agent in agents:
            if self.id in agent.arm_intervals:
                length = agent.arm_intervals[self.id][1] - agent.arm_intervals[self.id][0]
                mean   = agent.arm_means[self.id]

                total_episode_reward += mean * length
                total_episode_counts += length

                sampling_intervals.append(agent.arm_intervals[self.id])
                earliest_obs = min(earliest_obs, agent.arm_intervals[self.id][0])
                latest_obs   = max(latest_obs, agent.arm_intervals[self.id][1])

        for i in range(earliest_obs, latest_obs):
            for interval in sampling_intervals:
                if i >= interval[0] and i < interval[1]:
                    total_episode_pulls += 1
                    break 

        self.num_pulls     += total_episode_pulls
        self.num_samples   += total_episode_counts
        self.total_reward  += total_episode_reward
        self.estimated_mean = self.total_reward / self.num_samples
        self.conf_radius    = np.sqrt(2 * np.log(time) / self.num_pulls)
        self.ucb            = self.estimated_mean + self.conf_radius

    """BEGIN HACK"""
    def update_attributes_hack(self):
        self.num_pulls      = 1
        self.num_samples    = 1
        self.total_reward   = self.get_reward()
        self.estimated_mean = self.total_reward / self.num_samples
        self.conf_radius    = np.sqrt(2 * np.log(1) / self.num_pulls)
        self.ucb            = self.estimated_mean + self.conf_radius
    """END HACK"""

    def pull_individual(self, time, agents, agent_ids):
        single_reward = self.get_reward()
        reward = self.interaction.function(len(agent_ids)) * single_reward
        for a_id in agent_ids:
            agents[a_id].num_pulls_dict[self.id] += 1
            agents[a_id].total_reward_dict[self.id] += single_reward
            agents[a_id].estimated_mean_dict[self.id] = agents[a_id].total_reward_dict[self.id] / agents[a_id].num_pulls_dict[self.id]
            agents[a_id].conf_radius_dict[self.id] = np.sqrt(2 * np.log(time) / agents[a_id].num_pulls_dict[self.id])
            agents[a_id].ucb_dict[self.id] = agents[a_id].estimated_mean_dict[self.id] + agents[a_id].conf_radius_dict[self.id]

        return reward
    
    def __str__(self):
        return f"Arm {self.id}: True Mean = {self.true_mean}, Estimated Mean = {self.estimated_mean}, UCB = {self.ucb}, Num Pulls = {self.num_pulls}, Total Reward = {self.total_reward}"
    
    def reset(self):
        self.num_pulls = 0
        self.total_reward = 0
        self.estimated_mean = 0
        self.conf_radius = 0
        self.ucb = 0