import gurobipy as gp

def optimal_distribution(arm_list, M, theoretical = False):
    ### DOUBLE CHECK THE CHANGE TO THE F FUNCTIONS
    """
        Calculates the optimal distribution of agents over the arms.
        If theoretical == False, uses the current UCB estimates of each arm.
        If theoretical == True, uses the true means of each arm.
    """
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    store_vars = {}
    # This is the number of agents selecting each arm
    for arm in arm_list:
        store_vars = arm.interaction.add_constraints(m, store_vars)

    # Constraint that we can only pick M times
    m.addConstr(sum([store_vars[f"x_{arm.id}"] for arm in arm_list]) == M)
    if not theoretical:
        m.setObjective(sum([arm.ucb * store_vars[f"f(x_{arm.id})"] for arm in arm_list]), gp.GRB.MAXIMIZE)
    if theoretical:
        m.setObjective(sum([arm.true_mean * store_vars[f"f(x_{arm.id})"] for arm in arm_list]), gp.GRB.MAXIMIZE)

    m.optimize()
    if m.status == gp.GRB.INFEASIBLE:
        m.computeIIS()
        m.write("model.lp")
        m.write("model.ilp")

    store_values = m.getAttr("X", store_vars)
    return store_values, m.getObjective().getValue()