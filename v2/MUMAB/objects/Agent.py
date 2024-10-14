import random
import numpy as np
from numpy.random import normal as nl


class Agent:
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
        std_dev:              float, std_dev for sensor noise
        bias:                 float, bias for sensor noise
        move_prob:            float, probability of succesfully moving to a new node
        sample_prob:          float, probability of sampling.
        move_gamma:           float, decay rate of move probability
        sample_gamma:         float, decay rate of sample probability
        num_sample_failures:  int, number of times the agent has failed to sample
        num_move_failures:    int, number of times the agent has failed to move
        path:                 list, list of nodes that the agent must traverse to reach its target node
        G:                    networkX graph, the state graph
        arm_list:             list, list of arms that the agent has pulled during that episode
        reward_list:          list, list of rewards that the agent has received during that episode
        arm_intervals:        dict, dictionary with keys the arms. Items are [a, b] where a is the first time step during which the agent pulled the arm and b is the first time step for which the agent pulled a different arm
        arm_means:            dict, dictionary with keys the arms. Items are the mean rewards the agent recieved from that arm (ignoring sampling failures)
    Methods:
        move:           moves the agent to the inputted node
    """

    def __init__(
        self,
        id,
        node,
        G,
        std_dev,
        bias,
        move_prob,
        sample_prob,
        move_gamma,
        sample_gamma,
    ):
        # Agent attributes
        self.id: int = id
        self.current_node: dict = node
        self.num_pulls_dict = {i: 0 for i in G}
        self.total_reward_dict = {i: 0 for i in G}
        self.estimated_mean_dict = {}
        self.conf_radius_dict = {}
        self.ucb_dict = {}

        # Robustification attributes
        self.std_dev = std_dev
        self.bias = bias

        self.move_prob = move_prob
        self.move_gamma = move_gamma

        self.sample_prob = sample_prob
        self.sample_gamma = sample_gamma

        self.num_sample_failures = 0
        self.num_move_failures = 0
        
        self.sample_req = 0
        self.episode_sample_count = 0

        self.path = []

        self.G = G

        # Variables for communication protocol
        self.arm_list = []  # Keeps track of the visited arm throughout episode
        self.reward_list = []  # Keeps track of all rewards throughout the episode

        self.arm_intervals = {}  # Intervals sent to centralizer
        self.arm_means = {}  # Means sent to centralizer

    def move(self):
        if len(self.path) == 0:
            return

        if random.random() < self.move_prob * (self.move_gamma**self.num_move_failures):
            self.current_node = self.G.nodes[self.path[0]]
            self.path = self.path[1:]

    def at_target_pose(self):
        return len(self.path) == 0

    def get_current_node(self):
        return self.current_node

    def set_target_path(self, path):
        self.path = path

    def get_path_len(self):
        return len(self.path)

    def sample(self, true):
        """
        Return a noisy observation of the true reward. If the agent fails to sample, return None
        The sample probability is alpha * gamma^n where
            alpha is initial probability,
            gamma is the growth rate,
            n is the number of failures
        """
        if random.random() < self.sample_prob * (
            self.sample_gamma**self.num_sample_failures
        ):
            sample = true + np.clip(nl(self.bias, scale=self.std_dev), -1, 1)
        else:
            self.num_sample_failures += 1
            sample = None

        self.arm_list.append(self.current_node["arm"].id)
        self.reward_list.append(sample)
        self.episode_sample_count += 1

    def define_package(self):
        assert self.arm_intervals == {}
        assert self.arm_means == {}

        # Start by recreating intervals
        for i, arm in enumerate(self.arm_list):
            if i == 0 or arm != self.arm_list[i - 1]:
                # This is a new arm. Should not have already been visited
                try:
                    assert arm not in self.arm_intervals
                except:
                    print(self.arm_intervals)
                    print(arm)
                    print(self.arm_list)
                self.arm_intervals[arm] = [i, -1]

                if i > 0:
                    # Previous arm should exist and not have updated end time
                    assert arm != self.arm_list[i - 1]
                    assert self.arm_intervals[self.arm_list[i - 1]][1] == -1
                    self.arm_intervals[self.arm_list[i - 1]][1] = i

        # Update end time for last arm
        self.arm_intervals[self.arm_list[-1]][1] = len(self.arm_list)

        # Convert to np.array; this also converts Nones to nans
        self.reward_list = np.array(self.reward_list, dtype=float)

        # Now calculate means
        for arm in self.arm_intervals:
            self.arm_means[arm] = np.nanmean(
                self.reward_list[
                    self.arm_intervals[arm][0] : self.arm_intervals[arm][1]
                ]
            )

    def reset_package(self):
        self.arm_intervals = {}
        self.arm_means = {}
        self.arm_list = []
        self.reward_list = []

    def sampled_episode_req(self):
        return self.episode_sample_count >= self.sample_req

    def set_sample_req(self, sample_req):
        self.episode_sample_count = 0
        self.sample_req = sample_req