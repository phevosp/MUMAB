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
        move_alpha:           float, associated with exponential decay of sampling success probability
        move_beta:            float, associated with exponential decay of sampling success probability
        sample_prob:          float, probability of sampling succesfully
        sample_alpha:         float, associated with exponential decay of sampling success probability
        sample_beta:          float, associated with exponential decay of sampling success probability
        num_sample_failures:  int, number of times the agent has failed to sample
        num_move_failures:    int, number of times the agent has failed to move
        pull_req:             int, number of pulls (succesful & unsuccesful samples) required to complete an episode
        episode_pull_count:   int, number of pulls that the agent has completed in the current episode
        path:                 list, list of nodes that the agent must traverse to reach its target node
        G:                    networkX graph, the state graph
        arm_list:             list, list of arms that the agent has pulled during that episode
        reward_list:          list, list of rewards that the agent has received during that episode
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
        move_alpha,
        move_beta,
        sample_alpha,
        sample_beta,
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
        self.move_alpha = move_alpha
        self.move_beta = move_beta

        self.sample_prob = sample_prob
        self.sample_alpha = sample_alpha
        self.sample_beta = sample_beta

        self.num_sample_failures = 0
        self.num_move_failures = 0

        self.pull_req = 0
        self.episode_pull_count = 0

        self.path = []

        self.G = G

        # Variables for communication protocol
        self.arm_list = []  # Keeps track of the visited arm throughout episode
        self.reward_list = []  # Keeps track of all rewards throughout the episode

        # Variables specific to UCRL2 implementation
        self.policy = {}

    def move(self):
        if len(self.path) == 0:
            return
        move_prob = self.move_prob
        if self.move_alpha is not None:
            move_prob = self.move_prob * np.exp(
                -((self.num_move_failures / self.move_alpha) ** self.move_beta)
            )
        if random.random() < move_prob:
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
        # Note and update probability of succesful sample
        sample_prob = self.sample_prob
        if self.sample_alpha is not None:
            sample_prob = self.sample_prob * np.exp(
                -((self.num_sample_failures / self.sample_alpha) ** self.sample_beta)
            )
        if random.random() < sample_prob:
            sample = true + np.clip(nl(self.bias, scale=self.std_dev), -0.5, 0.5)
        else:
            self.num_sample_failures += 1
            sample = None

        self.arm_list.append(self.current_node["arm"].id)
        self.reward_list.append(sample)

        # For individial MAB algorithm
        if self.at_target_pose():
            self.episode_pull_count += 1

    def define_package(self):
        # Convert to np.array; this also converts Nones to nans
        self.reward_list = np.array(self.reward_list, dtype=float)

    def reset_package(self):
        self.arm_list = []
        self.reward_list = []

    def episode_pulls_req_met(self):
        # For individual MAB algorithm
        return self.episode_pull_count >= self.pull_req

    def set_episode_pulls_req(self, pull_req):
        # For individual MAB algorithm
        self.episode_pull_count = 0
        self.pull_req = pull_req

    def to_initialize(self):
        # For individual MAB algorithm
        # G will be G_indv so arms will be ArmIndividual
        return {
            node
            for node in self.G.nodes
            if (
                node["arm"].Arms[self.id].num_pulls == 0
                and node["arm"].Arms[self.id].episode_pulls == 0
            )
        }

    def set_policy(self, policy):
        # For UCRL2
        self.policy = policy

    def move_via_policy(self):
        # For UCRL2
        assert self.policy
        self.current_node = self.G.nodes[self.policy[self.current_node["id"]]]
