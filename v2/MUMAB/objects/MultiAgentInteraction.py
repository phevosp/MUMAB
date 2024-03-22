from abc import ABC, abstractmethod
import numpy as np
import gurobipy as gp


class MultiAgentInteractionInterface(ABC):

    @abstractmethod
    def function(self, x):
        return x

    @abstractmethod
    def add_constraints(self, m, store_vars, arm):
        pass

class ConcaveMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M, alpha=0.05):
        self.arm_id    :int = id
        self.log_base  :int = 2 + (id%10)
        self.alpha     :int = alpha
        self.M         :int = M

    def function(self, x):
        return (np.emath.logn(self.log_base, self.alpha*x + 1/(self.log_base)) + 1) / (np.emath.logn(self.log_base, self.alpha + 1/(self.log_base)) + 1)
    
    def add_constraints(self, m, store_vars):
        # This is the number of agents selecting each arm (call it x)
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # This is alpha*x_{} + 1/(log_base)
        temp1                                = m.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0.0, name = f"self.alpha*x_{self.arm_id}+1/(log_base)")
        # This is np.emath.logn(log_base, alpha*x + 1/(log_base))
        temp2                                = m.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = f"(np.emath.logn(log_base,self.alpha*x_{self.arm_id}+1/(log_base)))")
        # This is f(x) = (np.emath.logn(log_base, alpha*x + 1/(log_base)) + 1) / (np.emath.logn(log_base, alpha + 1/(log_base)) + 1)
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")


        # Add constraints
        m.addConstr(temp1 == self.alpha * store_vars[f"x_{self.arm_id}"] + 1/self.log_base, name = f"constr1_x_{self.arm_id}")
        m.addGenConstrLogA(temp1, temp2, self.log_base)
        m.addConstr(store_vars[f"f(x_{self.arm_id})"] == (temp2+1)/(np.emath.logn(self.log_base, self.alpha + 1/self.log_base) + 1), name = f"constr2_x_{self.arm_id}")

        return store_vars

class LinearMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M):
        self.arm_id    :int = id
        self.M         :int = M

    def function(self, x):
        return x
    
    def add_constraints(self, m, store_vars):
        # Number of agents at arm arm_id
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # Transformation function
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")

        # Constrain transformation == number of agents
        m.addConstr(store_vars[f"f(x_{self.arm_id})"] == store_vars[f"x_{self.arm_id}"], name = f"constr_x_{self.arm_id}")

        return store_vars
    
class ConstantMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M):
        self.arm_id    :int = id
        self.M         :int = M

    def function(self, x):
        if x > 0 :
            return 1
        else:
            return x
    
    def add_constraints(self, m, store_vars):

        # Number of agents at specific arm--denote by x
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # Transformation f(c)
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")

        # Indicator of x > 0
        b = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{self.arm_id} > 0")
       
        # Add constraints
        eps = 0.0001
        M = self.M + eps
        # bigM_constr1:
        # If b = 0, this is non-binding
        # If b = 1, this means x >= 0.001 -> x >= 1 because x is an integer
        m.addConstr(store_vars[f"x_{self.arm_id}"] >= eps - M * (1 - b), name="bigM_constr1")
        # bigM_constr2:
        # If b = 0, x <= 0, so x = 0
        # If b = 1, this is non-binding
        m.addConstr(store_vars[f"x_{self.arm_id}"] <= M * b, name="bigM_constr2")

        # If b == 1, then constrain f(x) == 1
        # If b == 0, then constrain f(x) == x == 0
        m.addConstr((b == 1) >> (store_vars[f"f(x_{self.arm_id})"] == 1), name = f"constr_x_{self.arm_id}_case_1")
        m.addConstr((b == 0) >> (store_vars[f"f(x_{self.arm_id})"] == store_vars[f"x_{self.arm_id}"]), name = f"constr_x_{self.arm_id}_case_2")

        return store_vars



class MuchMoreConcaveMultiAgentInteraction(ConcaveMultiAgentInteraction):
    def __init__(self, id, M, log_base_shift = 98):
        # Call concave, but shift log base by log_base_shift
        super().__init__(id, M)
        super().log_base = log_base_shift + super().id

    def function(self, x):
        super().function(x)

    def add_constraints(self, m, store_vars):
        super().add_constraints(m, store_vars)
class CollisionMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M):
        self.arm_id    :int = id
        self.M         :int = M

    def function(self, x):
        return x if x <= 1 else 0
    
    def add_constraints(self, m, store_vars):
        # This is the number of agents selecting each arm (call it x)
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # This is f(c)
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")
        # Indicator x > 1
        b = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{self.arm_id} > 1")
        # Constants

        # M is chosen to be as small as possible given the bounds on x and y
        eps = 0.0001
        M = self.M + eps
        # If b = 1, x > 1 (so x >= 2)
        # If b = 0, this is non-binding
        m.addConstr(store_vars[f"x_{self.arm_id}"] >= 1 + eps - M * (1 - b), name="bigM_constr1")
        # If b = 0, then x <= 1
        # If b = 1, then this is non-binding
        m.addConstr(store_vars[f"x_{self.arm_id}"] <= 1 + M * b, name="bigM_constr2")
        
        # Add constraints
        m.addConstr((b == 1) >> (store_vars[f"f(x_{self.arm_id})"] == 0), name = f"constr_x_{self.arm_id}_case_1")
        m.addConstr((b == 0) >> (store_vars[f"f(x_{self.arm_id})"] == store_vars[f"x_{self.arm_id}"]), name = f"constr_x_{self.arm_id}_case_2")
        return store_vars


def getFunction(id, function_type, params):
    if function_type == 'concave':
        return ConcaveMultiAgentInteraction(id, params.M)
    elif function_type == 'collision':
        return CollisionMultiAgentInteraction(id, params.M)
    elif function_type == 'more_concave':
        return MuchMoreConcaveMultiAgentInteraction(id, params.M)
    elif function_type == 'linear':
        return LinearMultiAgentInteraction(id, params.M)
    elif function_type == 'constant':
        return ConstantMultiAgentInteraction(id, params.M)
    else:
        return None