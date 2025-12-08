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
        breakpoints:    int, number of breakpoints for non-stationary rewards
        true_mean:      float, true mean of the arm
        pulls:          list, each entry is the number of times the arm was pulled at that time step
        samples:        list, each entry is the number of successful samples at that time step
        rewards:        list, each entry is the total reward accumulated at that time step
        num_pulls:      int, total number of pulls of the arm
        num_samples:    int, total number of successful samples of the arm
        total_reward:   float, total reward accumulated from the arm
        estimated_mean: float, estimated mean of the arm
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

    def __init__(self, id, interaction, T, breakpoints):
        self.id: int = id
        # Reward
        self.breakpoints: int = breakpoints
        self.true_mean: list = [random.random() * 0.75 for _ in range(breakpoints)]
        # Statistics
        self.pulls: list = [0 for _ in range(T)]  # Attempted samples
        self.samples: list = [0 for _ in range(T)]  # Successful samples
        self.rewards: list = [0 for _ in range(T)]  # Rewards
        # Attributes
        self.num_pulls: int = 0
        self.num_samples: int = 0
        self.total_reward: float = 0
        self.estimated_mean: float = 0
        self.conf_radius: float = 0
        self.ucb: float = 0
        # Interaction Function
        self.interaction: MultiAgentInteractionInterface = interaction
        # Episode Statistics
        self.episode_pulls: int = 0
        self.episode_pulls_req: int = 0

    def get_reward(self, curr_time, T):
        """Returns reward of the arm as sampled by the agent

        Args:
            curr_time (int): Current time step for breakpoint calculation. Defaults to None.
            T (int): Total time steps for breakpoint calculation. Defaults to None.

        Returns:
            float: The reward sampled from the arm.
        """
        index = (curr_time - 1) * self.breakpoints // T
        mean = self.true_mean[index]

        return np.clip(np.random.normal(loc=mean, scale=0.1), 0, 1)

    def pull(self, time, T):
        single_reward = self.get_reward(time, T)
        self.pulls[time - 1] += 1
        self.episode_pulls += 1  # Add to episode pull counter
        return single_reward

    def update_attributes(self, agents, time, K=None, sw=None, df=None):
        """
        Update the arm's attributes with the communication protocol
        """
        ep_len = len(agents[0].arm_list)
        for i in range(ep_len):
            for agent in agents:
                if agent.arm_list[i] == self.id:
                    if not math.isnan(agent.reward_list[i]):
                        sample_time = time - (ep_len - i)
                        self.rewards[sample_time] += agent.reward_list[i]
                        self.samples[sample_time] += 1

        # For sliding window, only consider last sw samples
        if sw:
            # Get statistics
            start_time = max(0, time - sw)
            self.num_pulls = sum(self.pulls[start_time:time])
            self.num_samples = sum(self.samples[start_time:time])
            self.total_reward = sum(self.rewards[start_time:time])

            # Calculate mean and confidence radius
            self.estimated_mean = (
                self.total_reward / self.num_samples if self.num_samples > 0 else 0
            )
            self.conf_radius = (
                np.sqrt(2 * np.log(1 / (1 - df)) / self.num_samples)
                if self.num_samples > 0
                else 1000000000000
            )

        # For discount factor, weight recent samples more
        elif df:
            # Get statistics
            if self.total_reward:
                discounts = [df ** (time - t - 1) for t in range(time - ep_len, time)]

                self.total_reward *= df ** (ep_len)
                self.total_reward += sum(
                    p * d
                    for p, d in zip(
                        self.rewards[time - ep_len - 1 : time - 1], discounts
                    )
                )

                self.num_pulls *= df ** (ep_len)
                self.num_pulls += sum(
                    p * d
                    for p, d in zip(self.pulls[time - ep_len - 1 : time - 1], discounts)
                )

                self.num_samples *= df ** (ep_len)
                self.num_samples += sum(
                    s * d
                    for s, d in zip(
                        self.samples[time - ep_len - 1 : time - 1], discounts
                    )
                )
            else:
                discounts = [df ** (time - t - 1) for t in range(time)]
                self.num_pulls = sum(
                    p * d for p, d in zip(self.pulls[:time], discounts)
                )
                self.num_samples = sum(
                    s * d for s, d in zip(self.samples[:time], discounts)
                )
                self.total_reward = sum(
                    r * d for r, d in zip(self.rewards[:time], discounts)
                )

            # Calculate mean and confidence radius
            self.estimated_mean = self.total_reward / self.num_samples
            self.conf_radius = np.sqrt(2 * np.log(1 / (1 - df)) / self.num_samples)

        # For all other algorithms, consider all samples
        else:
            # Get statistics
            self.num_pulls = sum(self.pulls[:time])
            self.num_samples = sum(self.samples[:time])
            self.total_reward = sum(self.rewards[:time])

            # Calculate mean and confidence radius
            self.estimated_mean = self.total_reward / self.num_samples
            self.conf_radius = np.sqrt(2 * np.log(time) / self.num_samples)

        # UCB Calculation
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

    def __str__(self):
        return f"Arm {self.id}: True Mean = {self.true_mean}, Estimated Mean = {self.estimated_mean}, UCB = {self.ucb}, Num Pulls = {self.num_pulls}, Total Reward = {self.total_reward}"

    def reset(self):
        self.num_pulls = 0
        self.total_reward = 0
        self.estimated_mean = 0
        self.conf_radius = 0
        self.ucb = 0
        self.num_samples = 0
        self.rewards = [0 for _ in range(len(self.rewards))]
        self.samples = [0 for _ in range(len(self.samples))]
        self.pulls = [0 for _ in range(len(self.pulls))]

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

    def reset(self):
        for arm in self.Arms:
            arm.reset()
