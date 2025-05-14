import torch
import random
from typing import Dict, Set, List, Tuple, Optional, Any
from tqdm.auto import tqdm
import logging


from GroupInterventionCBM.grouped_interventions_ilp import get_interventions

def random_concepts(
        global_intervention_mask: torch.Tensor,
        concept_info: Dict[str, Dict[str, Any]],
        concept_names: list,
        budget: float,
        include_uncertain: bool,
        concept_uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
    """Extract random concepts within a budget for multiple samples."""

    final_mask = global_intervention_mask.clone()
    n_samples, n_concepts = final_mask.shape
    revealed_concept_names_set: Set[str] = set()

    if not include_uncertain and concept_uncertainty is None:
        raise ValueError("concept_uncertainty tensor is required when include_uncertain is False.")
    
    # iterate over samples
    all_indices = list(range(n_concepts))
    for i in range(n_samples):
        step_intervention_mask = torch.zeros(n_concepts, dtype=torch.bool)
        paid_groups: Set[str] = set()
        remaining_budget = budget

        available_indices = [j for j in all_indices if not final_mask[i, j]]
        if not include_uncertain:
            available_indices = [j for j in available_indices if not concept_uncertainty[i, j]]
        if not available_indices:
            # all concepts are already selected or uncertain
            continue

        random.shuffle(available_indices)       # for random selection
        for concept_idx in available_indices:
            if remaining_budget <= 0.0: break
            concept_name = concept_names[concept_idx]
            info = concept_info[concept_name]
            c_marginal_cost = info['marginal_cost']
            c_setup_cost = info['setup_cost']
            c_group = info['group']
            
            # determine cost of the concept acquisition
            current_concept_cost = c_marginal_cost
            needs_setup_payment = c_group not in paid_groups
            if needs_setup_payment:
                current_concept_cost += c_setup_cost

            # select concept only if it fits remaining budget
            if current_concept_cost <= remaining_budget:
                step_intervention_mask[concept_idx] = True
                revealed_concept_names_set.add(concept_name)
                remaining_budget -= current_concept_cost
                if needs_setup_payment:
                    paid_groups.add(c_group)

        final_mask[i] = torch.logical_or(final_mask[i], step_intervention_mask)

    union_of_revealed_concepts = sorted(list(revealed_concept_names_set))
    return final_mask, union_of_revealed_concepts


def random_groups_within_budget(
        global_intervention_mask: torch.Tensor,
        structured_costs: Dict[str, Dict[str, Any]], 
        concept_names: list,
        budget: float,
        include_uncertain: bool,
        concept_uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
    """Selects random groups within a budget for multiple samples."""

    final_mask = global_intervention_mask.clone()
    n_samples, n_concepts = final_mask.shape
    revealed_concept_names_set: Set[str] = set()
    concept_name_to_index = {name: idx for idx, name in enumerate(concept_names)} # map concept names to indices
    
    if not include_uncertain and concept_uncertainty is None:
        raise ValueError("concept_uncertainty tensor is required when include_uncertain is False.")
    
    # iterate over samples
    for i in range(n_samples):
        step_intervention_mask = torch.zeros(n_concepts, dtype=torch.bool)
        step_selected_groups: Set[str] = set()
        remaining_budget = budget

        # filter concepts and randomize their order within groups for second phase
        filtered_costs_sample = {}
        for group, info in structured_costs.items():
            group_concepts = {}
            # filter concepts
            for concept_name, marginal_cost in info.get('concepts', {}).items():
                if concept_name not in concept_name_to_index: continue # Skip if concept name not in list
                concept_idx = concept_name_to_index[concept_name]
                is_uncertain = concept_uncertainty is not None and concept_uncertainty[i, concept_idx]
                if include_uncertain or not is_uncertain:
                    group_concepts[concept_name] = marginal_cost
            # randomize
            if group_concepts:
                randomized_concepts = list(group_concepts.items())
                random.shuffle(randomized_concepts)
                filtered_costs_sample[group] = {
                    'setup_cost': info.get('setup_cost', 0.0),
                    'concepts': dict(randomized_concepts)}

        # randomize group order
        available_groups_order = list(filtered_costs_sample.keys())
        random.shuffle(available_groups_order)

        # phase 1: select entire groups
        for group in available_groups_order:
            if remaining_budget <= 0.0: break
            group_info = filtered_costs_sample[group]
            group_setup_cost = group_info['setup_cost']
            current_group_marginal_cost = 0.0
            concepts_to_reveal_in_group = []

            # compute cost of acquiring group excluding already revealed concepts
            for concept_name, marginal_cost in group_info['concepts'].items():
                concept_idx = concept_name_to_index[concept_name]
                if not final_mask[i, concept_idx] and not step_intervention_mask[concept_idx]:
                    current_group_marginal_cost += marginal_cost
                    concepts_to_reveal_in_group.append((concept_name, concept_idx))
            total_group_cost = group_setup_cost + current_group_marginal_cost

            # select group only if it fits remaining budget
            if total_group_cost <= remaining_budget and concepts_to_reveal_in_group:
                remaining_budget -= total_group_cost
                step_selected_groups.add(group)
                for concept_name, concept_idx in concepts_to_reveal_in_group:
                    step_intervention_mask[concept_idx] = True
                    revealed_concept_names_set.add(concept_name)

        # phase 2: no entire groups fit, select individual concepts from the next random group
        if remaining_budget > 0.0:
            groups_order_p2 = [g for g in available_groups_order if g not in step_selected_groups] # No need to re-shuffle, use the existing random order
            
            # select eligible groups, i.e., groupd with setup cost <= remaining budget
            # and with 1+ unrevealed concepts
            eligible_groups = []
            for group in groups_order_p2:
                group_info = filtered_costs_sample[group]
                if group_info['setup_cost'] <= remaining_budget:
                    has_unrevealed_concept = False
                    for concept_name in group_info['concepts']:
                         concept_idx = concept_name_to_index[concept_name]
                         if not final_mask[i, concept_idx] and not step_intervention_mask[concept_idx]:
                             has_unrevealed_concept = True
                             break
                    if has_unrevealed_concept:
                         eligible_groups.append(group)

            # select individual concepts from each eligible group, sequentially by group
            if eligible_groups:
                setup_paid_p2: Set[str] = set()
                for group in eligible_groups: 
                    if remaining_budget <= 0.0: break
                    group_info = filtered_costs_sample[group]
                    setup_cost = group_info['setup_cost']
                    needs_setup_payment = (group not in setup_paid_p2)

                    for concept_name, marginal_cost in group_info['concepts'].items():
                        if remaining_budget <= 0.0: break
                        concept_idx = concept_name_to_index[concept_name]

                        if not final_mask[i, concept_idx] and not step_intervention_mask[concept_idx]:
                            # potential cost of the concept
                            current_concept_cost = marginal_cost
                            current_setup_cost_to_pay = 0.0
                            if needs_setup_payment:
                                current_setup_cost_to_pay = setup_cost
                            total_cost = current_concept_cost + current_setup_cost_to_pay

                            # add only if it fits
                            if total_cost <= remaining_budget:
                                remaining_budget -= total_cost
                                step_intervention_mask[concept_idx] = True
                                revealed_concept_names_set.add(concept_name)
                                step_selected_groups.add(group) 

                                if needs_setup_payment and current_setup_cost_to_pay > 0:
                                    setup_paid_p2.add(group)
                                    needs_setup_payment = False
                                elif needs_setup_payment: 
                                    needs_setup_payment = False

        final_mask[i] = torch.logical_or(final_mask[i], step_intervention_mask)

    union_of_revealed_concepts = sorted(list(revealed_concept_names_set))
    return final_mask, union_of_revealed_concepts

def greedy_concepts(
        global_intervention_mask: torch.Tensor, 
        concept_info: Dict[str, Dict[str, Any]],
        concept_values: torch.Tensor, 
        concept_names: List[str],
        budget: float,
        include_uncertain: bool,
        concept_uncertainty: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts the top-k concepts within a budget until budget is exhausted.
    Ensures that the group cost is payed only once if 2+ concepts comes from the same group.
    """
    final_mask = global_intervention_mask.clone()
    n_samples, n_concepts = final_mask.shape
    revealed_concept_names_set: Set[str] = set()

    if not include_uncertain and concept_uncertainty is None:
        raise ValueError("concept_uncertainty tensor is required when include_uncertain is False.")
    
    # iterate over samples
    for i in range(n_samples):
        step_intervention_mask = torch.zeros(n_concepts, dtype=torch.bool)
        paid_groups: Set[str] = set()
        remaining_budget = budget

        eligible_concepts = []
        for concept_idx in range(n_concepts): 
            if final_mask[i, concept_idx]: continue # concept already selected
            is_uncertain = not include_uncertain and concept_uncertainty is not None and concept_uncertainty[i, concept_idx]
            if is_uncertain: continue
            concept_name = concept_names[concept_idx]
            if concept_name not in concept_info: 
                logging.warning(f"Concept {concept_name} not found in concept_info.")
                continue # skip if concept name not in list 
            
            # add concept with its value to the list
            value = concept_values[i, concept_idx].item()
            eligible_concepts.append({'idx': concept_idx, 'name': concept_name, 'value': value})

        # sort candidates by value 
        sorted_candidates = sorted(eligible_concepts, key=lambda x: x['value'], reverse=True)
        if not sorted_candidates: continue 

        # greedy selection loop
        for candidate in sorted_candidates: 
            if remaining_budget <= 0.0: break
            concept_idx = candidate['idx']
            concept_name = candidate['name']
            if step_intervention_mask[concept_idx]: continue # should not happen

            # get cost info
            info = concept_info[concept_name]
            c_marginal_cost = info['marginal_cost']
            c_setup_cost = info['setup_cost']
            c_group = info['group'] 

            # compute cost of the concept acquisition
            current_concept_cost = c_marginal_cost
            needs_setup_payment = c_group not in paid_groups
            if needs_setup_payment:
                current_concept_cost += c_setup_cost
            
            # select concept only if it fits remaining budget
            if current_concept_cost <= remaining_budget:
                step_intervention_mask[concept_idx] = True
                revealed_concept_names_set.add(concept_name)
                remaining_budget -= current_concept_cost
                if needs_setup_payment:
                    paid_groups.add(c_group)
            
        final_mask[i] = torch.logical_or(final_mask[i], step_intervention_mask)

    union_of_revealed_concepts = sorted(list(revealed_concept_names_set))
    return final_mask, union_of_revealed_concepts

def greedy_groups(
        global_intervention_mask: torch.Tensor,
        structured_costs: Dict[str, Dict[str, Any]],
        concept_values: torch.Tensor,
        concept_names: List[str],
        budget: float,
        include_uncertain: bool,
        concept_uncertainty: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, List[str]]:
    """
    Extracts the top-k groups within a budget until budget is exhausted. Then extracts 
    top-k concepts from the remaining most-valuable group.
    """
    final_mask = global_intervention_mask.clone()
    n_samples, n_concepts = final_mask.shape
    revealed_concept_names_set: Set[str] = set()
    concept_name_to_index = {name: idx for idx, name in enumerate(concept_names)} # map concept names to indices

    if not include_uncertain and concept_uncertainty is None:
        raise ValueError("concept_uncertainty tensor is required when include_uncertain is False.")
    
    # iterate over samples
    for i in range(n_samples): 
        step_intervention_mask = torch.zeros(n_concepts, dtype=torch.bool)
        step_selected_groups: Set[str] = set()
        remaining_budget = budget
        groups_selected_p1 = set()

        # filter concepts and build eligible groups info 
        eligible_groups = []
        for group, info in structured_costs.items():
            group_setup_cost = info.get('setup_cost', 0.0)
            eligible_concepts_in_group = []
            group_value = 0.0 

            for concept_name, marginal_cost in info.get('concepts', {}).items():
                if concept_name not in concept_name_to_index: continue
                concept_idx = concept_name_to_index[concept_name]

                # check concept eligibility
                is_revealed_globally = final_mask[i, concept_idx]
                is_uncertain =  not include_uncertain and concept_uncertainty is not None and concept_uncertainty[i, concept_idx]

                # consider it to the group if eligible
                if not is_revealed_globally and not is_uncertain:
                    c_value = concept_values[i, concept_idx].item()
                    group_value += c_value
                    eligible_concepts_in_group.append({
                        'name': concept_name,
                        'idx': concept_idx,
                        'marginal_cost': marginal_cost,
                        'value': c_value})
                
            if eligible_concepts_in_group:
                eligible_groups.append({
                    'name': group,
                    'setup_cost': group_setup_cost,
                    'initial_value': group_value,
                    'eligible_concepts': eligible_concepts_in_group})
        

        # phase 1: select entire groups
        sorted_groups_p1 = sorted(eligible_groups, key=lambda x: x['initial_value'], reverse=True)

        for group_data in sorted_groups_p1:
            if remaining_budget <= 0.0: break

            group_name = group_data['name']
            group_setup_cost = group_data['setup_cost']
            current_dynamic_cost = 0.0
            concepts_to_acquire_now = []
            needs_setup_payment = group_name not in step_selected_groups
            if needs_setup_payment: current_dynamic_cost += group_setup_cost
            for concept_detail in group_data['eligible_concepts']:
                concept_idx = concept_detail['idx']
                if not step_intervention_mask[concept_idx]:
                    current_dynamic_cost += concept_detail['marginal_cost']
                    concepts_to_acquire_now.append(concept_detail)

            if concepts_to_acquire_now and current_dynamic_cost <= remaining_budget:
                remaining_budget -= current_dynamic_cost
                groups_selected_p1.add(group_name)
                if needs_setup_payment: step_selected_groups.add(group_name)
                for concept_detail in concepts_to_acquire_now:
                    concept_idx = concept_detail['idx']
                    step_intervention_mask[concept_idx] = True
                    revealed_concept_names_set.add(concept_detail['name'])

        # phase 2: no entire group fits, select individual concepts from the next most valuable group
        if remaining_budget > 0.0:
            eligible_concepts_p2 = []
            
            # prepare eligible groups and concepts for phase 2
            for group_data in eligible_groups:
                # no need to explicitly filter groups selected in p1, as the intervention mask will take care of it
                group_name = group_data['name']
                group_setup_cost = group_data['setup_cost']

                for concept_detail in group_data['eligible_concepts']:
                    concept_idx = concept_detail['idx']
                    if step_intervention_mask[concept_idx]: continue

                    value = concept_detail['value']

                    # compute cost
                    marginal_cost = concept_detail['marginal_cost']
                    current_concept_cost = marginal_cost
                    needs_setup_payment_check = group_name not in step_selected_groups
                    if needs_setup_payment_check:
                        current_concept_cost += group_setup_cost
                    # add if it fits
                    if current_concept_cost <= remaining_budget:
                        eligible_concepts_p2.append({
                            'name': concept_detail['name'], 
                            'idx': concept_idx,
                            'value': value, 
                            'group': group_name,
                            'setup_cost': group_setup_cost, 
                            'marginal_cost': marginal_cost
                         })
            
            # sort candidates by value
            sorted_concepts_p2 = sorted(eligible_concepts_p2, key=lambda x: x['value'], reverse=True)

            # select individual concepts from each eligible group, sequentially by group
            for concept_data in sorted_concepts_p2:
                if remaining_budget <= 0.0: break

                concept_idx = concept_data['idx']
                group_name = concept_data['group']

                # Re-check mask, as it might have changed during this loop
                if step_intervention_mask[concept_idx]: continue

                # Recalculate cost at the moment of selection
                current_cost = concept_data['marginal_cost']
                needs_setup_payment_now = group_name not in step_selected_groups
                if needs_setup_payment_now:
                    current_cost += concept_data['setup_cost']

                if current_cost <= remaining_budget:
                    remaining_budget -= current_cost
                    step_intervention_mask[concept_idx] = True
                    revealed_concept_names_set.add(concept_data['name'])
                    if needs_setup_payment_now:
                        step_selected_groups.add(group_name)

        final_mask[i] = torch.logical_or(final_mask[i], step_intervention_mask)

    union_of_revealed_concepts = sorted(list(revealed_concept_names_set))
    return final_mask, union_of_revealed_concepts


def optimized(
        global_intervention_mask: torch.Tensor, 
        structured_costs: Dict[str, Dict[str, Any]],
        concept_values: torch.Tensor, # shape (n_samples, n_concepts), value for each concept
        concept_names: List[str],
        budget: float,
        include_uncertain: bool,
        concept_uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
    """Select best concepts and groups within a budget using ILP."""

    final_mask = global_intervention_mask.clone()
    n_samples, n_concepts = final_mask.shape
    revealed_concept_names_set: Set[str] = set()
    concept_name_to_index = {name: idx for idx, name in enumerate(concept_names)}

    if not include_uncertain and concept_uncertainty is None:
        raise ValueError("concept_uncertainty tensor is required when include_uncertain is False.")
    
    # iterate over samples 
    for i in tqdm(range(n_samples), desc="Optimizing interventions", unit="sample", leave=False):
        # determine eligilble concepts 
        eligible_concepts_dict = {}
        for concept_idx in range(n_concepts):
            if final_mask[i, concept_idx]: continue
            is_uncertain = not include_uncertain and concept_uncertainty is not None and concept_uncertainty[i, concept_idx]
            if is_uncertain: continue

            concept_name = concept_names[concept_idx]
            value = concept_values[i, concept_idx].item()
            eligible_concepts_dict[concept_name] = {'value': value}
        
        if not eligible_concepts_dict: continue # skip if no eligible concepts

        # get the interventions for this sample (no need to filter structured_costs as it is handled in the function)
        _, selected_concepts_names, _ = get_interventions(concept_values=eligible_concepts_dict, structured_costs=structured_costs, budget=budget)
        
        if selected_concepts_names:
            selected_indices = [concept_name_to_index[name] for name in selected_concepts_names]
            update_indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
            if len(update_indices_tensor) > 0:
                final_mask[i, update_indices_tensor] = True
            revealed_concept_names_set.update(selected_concepts_names)
    
    union_of_revealed_concepts = sorted(list(revealed_concept_names_set))
    return final_mask, union_of_revealed_concepts
            

        