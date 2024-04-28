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
    Methods:
        move:           moves the agent to the inputted node
    """
    def __init__(self, id, node, G, std_dev, bias, move_prob, sample_prob):
        # Agent attributes
        self.id           :int  = id
        self.current_node :dict = node
        self.num_pulls_dict      = {i: 0 for i in G}
        self.total_reward_dict   = {i: 0 for i in G}
        self.estimated_mean_dict = {}
        self.conf_radius_dict    = {}
        self.ucb_dict            = {}
        self.std_dev             = std_dev
        self.bias                = bias   
        self.move_prob           = move_prob
        self.sample_prob         = sample_prob   

    # def update_move_prob():
        

    # def update_sample_prob():
        

    def move(self, new_node):
        self.current_node = new_node

    def observation(self, true):
        # Sample with probability sample_prob
        if random.random() < self.sample_prob:
            return nl(self.bias + true, self.std_dev, 1)[0]
        else:
            return None
