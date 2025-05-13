import pulp
import logging


def get_interventions(concept_values, structured_costs, budget): 
    """
    Parameteres
    ----------
    concept_values : dict
        Dictionary with concept values. Keys are concept ids, values are value of the related concept.
    structured_costs : dict
        Dictionary with structured costs. Keys are group ids, values are dictionaries with setup_cost and concepts.
        Concepts is a dictionary with concept ids as keys and their marginal costs as values.
    budget : float
        Budget for the selection process.
    
    Returns
    -------
    dict
        Dictionary with selected concepts. Keys are group ids, values are lists of selected concept ids.
    list
        List of selected concept ids.
    int
        Status code of the optimization process.
    
    Example
    ------- 
    concept_values = {
        'concept1': {'value': 10},
        'concept2': {'value': 20},
        'concept3': {'value': 30}
    }
    structured_costs = {
        'group1': {
            'setup_cost': 100,
            'concepts': {
                'concept1': 5,
                'concept2': 10
            }
        },
        'group2': {
            'setup_cost': 200,
            'concepts': {
                'concept2': 15,
                'concept3': 20
            }
        }
    }
    """
    def get_concept_cost(i, j):
        return structured_costs[j]['concepts'][i]

    # define model 
    model = pulp.LpProblem("Grouped_Concept_Selection_Phase1", pulp.LpMaximize)

    # Define decision variables, y[(i,j)] = 1 if concept i is selected in group j, z[j] = 1 if group j is selected
    y, z = {}, {}
    for j in structured_costs:
        z[j] = pulp.LpVariable(f'z_{j}', 0, 1, pulp.LpBinary)
        if isinstance(structured_costs[j].get('concepts'), dict):
            for i in structured_costs[j]['concepts']:
                if i in concept_values:
                    y[(i, j)] = pulp.LpVariable(f'y_{i}_{j}', 0, 1, pulp.LpBinary)
        else:
            raise ValueError(f"Set '{j}' does not have a valid 'concepts' dictionary")


    # objective function: maximize sum of selected concepts values
    model += pulp.lpSum(y[i, j] * concept_values[i]['value'] for (i, j) in y if i in concept_values), "Total_Value" # Check concepts[i]['value'] exists

    # constraint 1: budget
    setup_costs = pulp.lpSum(z[j] * structured_costs[j]['setup_cost'] for j in structured_costs if j in z) # Check sets[j]['setup_cost'] exists
    marginal_costs = pulp.lpSum(y[i, j] * get_concept_cost(i, j) for (i, j) in y) # Use the cost function
    model += (setup_costs + marginal_costs <= budget, "Budget_Constraint")

    # constraint 2: y_ij <= z_j. The concept i can only be selected from j if j is selected
    for (i, j) in y:
        if j in z:
            model += y[i, j] <= z[j], f"Linkage_Constraint_{i}_{j}"
        else:
            raise ValueError(f"Linkage constraint for y_{i}_{j} failed: z_{j} not found. This shouldn't happen.")
            
    # constraint 3: sum_j(y_ij) <= 1. Each concept i can only be selected from one group j
    concepts_in_problem = set(i for (i, j) in y)
    for i in concepts_in_problem:
        model += pulp.lpSum(y[i, j] for j in structured_costs if (i,j) in y) <= 1, f"Single_Selection_{i}"

    # solve 
    model.solve(pulp.PULP_CBC_CMD(msg=0)) 
    status1_code = model.status
    status1_text = pulp.LpStatus[status1_code]

    if status1_code != pulp.LpStatusOptimal:
        logging.error(f"\033[91mPhase 1 did not find an optimal solution (Status: {status1_text}). Cannot proceed to Phase 2. Returning empty selection.\033[0m")
        return {}, [], status1_code 
    if model.objective is None or pulp.value(model.objective) is None:
        logging.error(f"\033[91mPhase 1 status is Optimal, but objective value is missing. This is unexpected. Returning empty.\033[0m")
        return {}, [], status1_code

    select_concept_in_set_model1 = {j: [] for j in structured_costs}
    selected_concepts_model1 = set()
    phase1_cost = 0
    if status1_code == pulp.LpStatusOptimal: # Redundant check, but safe
        for j_idx in z:
             if z[j_idx].value() is not None and z[j_idx].value() > 0.5:
                 phase1_cost += structured_costs[j_idx]['setup_cost']
        for (i, j) in y:
            if y[i, j].value() is not None and y[i, j].value() > 0.5:
                select_concept_in_set_model1[j].append(i)
                selected_concepts_model1.add(i)
                phase1_cost += get_concept_cost(i,j) 

    final_selection_dict = {j: concepts_list for j, concepts_list in select_concept_in_set_model1.items() if concepts_list}
    return final_selection_dict, list(selected_concepts_model1), model