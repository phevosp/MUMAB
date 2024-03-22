import gurobipy as gp

def optimal_distribution(arm_list, M, theoretical=False, minimize=False, debug=False):
    """
        Calculates the optimal distribution of agents over the arms.
        If theoretical,  uses the true means of each arm.
        If not theoretical, uses the current UCB estimates of each arm.
        If minimize, will minimize; should only call this when theoretical = True
        If not minimize, will maximize
        If debug, will set output flag to 1 and write .lp
    """
    m = gp.Model("mip1")
    output_flag = 1 if debug else 0
    m.setParam('OutputFlag', output_flag)
    store_vars = {}
    for arm in arm_list:
        store_vars = arm.interaction.add_constraints(m, store_vars)

    # Constraint that we can only pick M times
    m.addConstr(sum([store_vars[f"x_{arm.id}"] for arm in arm_list]) == M)

    # Define objective function
    obj = gp.GRB.MINIMIZE if minimize else gp.GRB.MAXIMIZE
    if not theoretical:
        m.setObjective(sum([arm.ucb * store_vars[f"f(x_{arm.id})"] for arm in arm_list]), obj)
    if theoretical:
        m.setObjective(sum([arm.true_mean * store_vars[f"f(x_{arm.id})"] for arm in arm_list]), obj)

    m.optimize()

    # Print model.lp
    model_name = 'model_minimize' if minimize else 'model_maximize'
    if m.status == gp.GRB.INFEASIBLE:
        m.computeIIS()
        m.write(f"{model_name}.lp")
        m.write(f"{model_name}.ilp")
    elif debug:
        m.write(f"{model_name}.lp")

    store_values = m.getAttr("X", store_vars)
    return store_values, m.getObjective().getValue()