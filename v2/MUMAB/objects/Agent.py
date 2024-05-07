import random
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
        dynamic_prob:         bool, whether the agent updates its move probability and sample probability based on its success rate
    Methods:
        move:           moves the agent to the inputted node
    """
    def __init__(self, id, node, G, std_dev, bias, move_prob, sample_prob, move_gamma, sample_gamma):
        # Agent attributes
        self.id           :int  = id
        self.current_node :dict = node
        self.num_pulls_dict      = {i: 0 for i in G}
        self.total_reward_dict   = {i: 0 for i in G}
        self.estimated_mean_dict = {}
        self.conf_radius_dict    = {}
        self.ucb_dict            = {}

        # Robustification attributes
        self.std_dev             = std_dev
        self.bias                = bias   

        self.move_prob           = move_prob
        self.move_gamma          = move_gamma

        self.sample_prob         = sample_prob
        self.sample_gamma        = sample_gamma 

        self.num_sample_failures = 0
        self.num_move_failures   = 0

        self.path                = []

        self.G                   = G

    def move(self):
        if len(self.path) == 0:
            return
        
        if random.random() < self.move_prob * (self.move_gamma**self.num_move_failures):
            self.current_node = self.G.nodes[self.path[0]]
            self.path = self.path[1:]

    def at_target_pose(self):
        return len(self.path) == 0
    
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
        if random.random() < self.sample_prob * (self.sample_gamma**self.num_sample_failures):
            return nl(self.bias + true, self.std_dev, 1)[0]
        else:
            self.num_sample_failures += 1
            return None
