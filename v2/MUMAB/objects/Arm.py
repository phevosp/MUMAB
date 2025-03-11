import numpy as np
import random
from MUMAB.objects.MultiAgentInteraction import MultiAgentInteractionInterface
import math
import copy


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
        interaction:    function, function used to compute multiplicative benefit of multiple agents sampling the same arm
        episode_pulls:  int, number of pulls that the arm has received in the current episode
        episode_pulls_req: int, number of pulls required to complete an episode

    Methods:
        get_reward:     returns random reward for pulling the arm
        pull:           updates the arm's attributes after simulating a pull of the arm
        reset:          resets the arm's attributes
    """

    def __init__(self, id, interaction, K):
        self.id: int = id
        self.true_mean: float = random.random() * 0.75
        self.num_pulls: int = 0  # Number of succesful or unsuccesful samples
        self.num_samples: int = 0  # Number of succesful samples
        self.total_reward: int = 0
        self.estimated_mean: int = 0
        self.conf_radius: int = 0
        self.ucb: int = 0
        self.interaction: MultiAgentInteractionInterface = interaction
        self.episode_pulls: int = 0
        self.episode_pulls_req: int = 0

    def get_reward(self):
        return np.clip(np.random.normal(loc=self.true_mean, scale=0.1), 0, 1)

    def pull(self, num_agents):
        single_reward = self.get_reward()
        self.episode_pulls += 1
        return single_reward

    def update_attributes(self, agents, time):
        """
        Update the arm's attributes with the communication protocol
        """
        total_episode_reward = 0
        total_episode_counts = 0

        for i in range(len(agents[0].arm_list)):
            for agent in agents:
                if agent.arm_list[i] == self.id:
                    if not math.isnan(agent.reward_list[i]):
                        total_episode_reward += agent.reward_list[i]
                        total_episode_counts += 1

        self.num_pulls += self.episode_pulls
        self.num_samples += total_episode_counts
        self.total_reward += total_episode_reward
        self.estimated_mean = self.total_reward / self.num_samples
        self.conf_radius = np.sqrt(2 * np.log(time) / self.num_samples)
        self.ucb = self.estimated_mean + self.conf_radius

    def update_attributes_UCRL2(self, agents, time, num_arms, num_edges, delta):
        total_episode_reward = 0
        total_episode_counts = 0

        for i in range(len(agents[0].arm_list)):
            for agent in agents:
                if agent.arm_list[i] == self.id:
                    if not math.isnan(agent.reward_list[i]):
                        total_episode_reward += agent.reward_list[i]
                        total_episode_counts += 1

        self.num_pulls += self.episode_pulls
        self.num_samples += total_episode_counts
        self.total_reward += total_episode_reward
        self.estimated_mean = self.total_reward / self.num_samples
        self.conf_radius = np.sqrt(
            7 * np.log(time * num_arms * num_edges / delta) / (2 * self.num_samples)
        )
        self.ucb = self.estimated_mean + self.conf_radius

    def update_attributes_hack(self, num_agents, type, num_arms, num_edges, delta):
        """
        Update the arm's attributes during the hack initialization phase
        Takes in agents and the type of algorithm (simple or robust or UCRL2)
        """
        self.num_pulls = 1
        self.num_samples = 1
        self.total_reward = self.get_reward()
        self.estimated_mean = self.total_reward / self.num_samples
        self.conf_radius = (
            np.sqrt(2 * np.log(1) / self.num_pulls)
            if type == "simple" or type == "robust"
            else np.sqrt(
                7 * np.log(1 * num_arms * num_edges / delta) / (2 * self.num_samples)
            )
        )
        self.ucb = self.estimated_mean + self.conf_radius

    def __str__(self):
        return f"Arm {self.id}: True Mean = {self.true_mean}, Estimated Mean = {self.estimated_mean}, UCB = {self.ucb}, Num Pulls = {self.num_pulls}, Total Reward = {self.total_reward}"

    def reset(self):
        self.num_pulls = 0
        self.total_reward = 0
        self.estimated_mean = 0
        self.conf_radius = 0
        self.ucb = 0
        self.num_samples = 0

    def set_episode_pulls_req(self, episode_pulls_req):
        self.episode_pulls = 0
        self.episode_pulls_req = episode_pulls_req

    def episode_pulls_req_met(self):
        return self.episode_pulls >= self.episode_pulls_req


class ArmIndividual:
    def __init__(self, Arm, M):
        self.Arms = [copy.deepcopy(Arm) for _ in range(M)]
        self.id = Arm.id

    def update_attributes(self, agent, time):
        self.Arms[agent.id].update_attributes([agent], time)

    def update_attributes_hack(self):
        for arm in self.Arms:
            arm.update_attributes_hack(1, "simple", 0, 0, 0)

    def reset(self):
        for arm in self.Arms:
            arm.reset()
