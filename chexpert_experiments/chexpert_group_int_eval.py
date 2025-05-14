import os
import sys
import time
import logging
from typing import Sequence, Dict, Any, Tuple, Optional, Callable, List
import datetime
import enum
import random 
import json
from functools import partial

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from absl import app
from absl import flags
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Bernoulli, kl
import torchmetrics

import network
from datasets import chexpert_dataset
import enum_utils
import train_util
import utils
from intervention_util import *
from GroupInterventionCBM import policies


random.seed(0) # for reproducibility, change if needed

#NOTE: we assume that the folders is which we save the models have specific paths
# change in the main code if needed

# change _BTYPE to switch from independent/joint bottleneck
_BTYPE = flags.DEFINE_enum_class('bottleneck_type', default=enum_utils.BottleneckType.INDEPENDENT, enum_class=enum_utils.BottleneckType, help='Type of bottleneck models used (determines how models are loaded).')
_ARCH = flags.DEFINE_enum_class('arch', default=enum_utils.Arch.X_TO_C_TO_Y, enum_class=enum_utils.Arch, help='Architecture to use for training.')
_NON_LINEAR_CTOY = flags.DEFINE_bool('non_linear_ctoy', default=False, help='Whether to use a non-linear CtoY model.')
_DATASET = flags.DEFINE_enum_class('dataset', default=enum_utils.Dataset.CHEXPERT, enum_class=enum_utils.Dataset, help='Dataset to use for training.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', default=32, help='Batch Size')
_MERGE_TRAIN_AND_VAL = flags.DEFINE_bool('merge_train_and_val', default=False, help='Whether to merge training and validation sets for training.')
_OPTIMIZER = flags.DEFINE_enum('optimizer', default='sgd', enum_values=['sgd', 'adam'], help='Optimizer to use for training.')
_LR = flags.DEFINE_float('lr', default=1e-3, help='Learning rate.')
_WD = flags.DEFINE_float('wd', default=0, help='Weight decay.')
_LOSS_WEIGHTS = flags.DEFINE_list( 'loss_weights', default=None, help='Loss weights')
_NUM_WORKERS = flags.DEFINE_integer('num_workers', default=4, help='Number of DataLoader workers.')
_LOADING_PATH = flags.DEFINE_string('loading_path', default='', help='Path to checkpoint file to load and resume training.')

# if you want to perform more than 1 batched intervention
_N_INTERVENTION_STEPS = flags.DEFINE_integer('n_intervention_steps', default=1, help='Number of intervention steps to perform.')

# budget for each step
_INTERVENTION_BUDGET = flags.DEFINE_integer('budget', default=30, help='Budget for intervention selection at each step (number of groups/concepts).')

# other intervention parameters
_CONCEPT_METRIC = flags.DEFINE_enum_class('concept_metric', default=enum_utils.Metric.CONCEPT_ENTROPY, enum_class=enum_utils.Metric, help='Metric used for concept uncertainty/selection.')
_LABEL_METRIC = flags.DEFINE_enum_class( 'label_metric', default=enum_utils.Metric.LABEL_CONFIDENCE_CHANGE, enum_class=enum_utils.Metric, help='Metric used for label-based concept importance.')
_LABEL_METRIC_WEIGHT = flags.DEFINE_float('label_metric_weight', default=0.5, help='Weighting factor for the label importance metric relative to the concept uncertainty metric.')
_INTERVENTION_FORMAT = flags.DEFINE_enum_class( 'intervention_format', default=enum_utils.InterventionFormat.BINARY, enum_class=enum_utils.InterventionFormat, help='Format for representing intervened concepts (LOGITS, PROBS, BINARY).')
_INCLUDE_UNCERTAIN_IN_INTERVENTION = flags.DEFINE_bool( 'include_uncertain_in_intervention', default=False, help='Whether to allow intervention on concepts marked as uncertain in the dataset.')

# policy to run (optimized, random_concepts, random_groups, greedy_concepts, greedy_groups)
_POLICY_TYPE = flags.DEFINE_string('policy_type', default='optimized', help='Type of policy to use for concept selection.')

# paths
_RESULTS_FILE = flags.DEFINE_string('results_file', default='intervention_results.json', help='File path to save the intervention results.')
_BASE_CHECKPOINT_DIR = flags.DEFINE_string('checkpoint_dir', default='./chexpert_experiments/results', help='Directory where checkpoints are saved')
_TRAINING_SEEDS = flags.DEFINE_list('training_seeds', default=[-1], help='List of training seeds to use for evaluation and access folders') # -1 for unset seed

FLAGS = flags.FLAGS


def _get_true_predicted_concepts_true_labels(xtoc_model: nn.Module, ds_val: DataLoader, device: torch.device):
    """Runs X->C model on policy_dataloader to get concept predictions. Returns true concepts, uncertainty, predicted concepts, and true labels."""
    true_concepts_list = []
    concept_uncertainty_list = []
    pred_concepts_list = []
    true_labels_list = []

    xtoc_model.eval()
    with torch.no_grad():
        for batch in ds_val:
            # Ensure batch unpacking matches dataset structure
            if len(batch) == 4:
                images, true_concepts, true_labels, concept_uncertainty = batch
            elif len(batch) == 3: # Assuming uncertainty might not always be present
                images, true_concepts, true_labels = batch
                concept_uncertainty = torch.zeros_like(true_concepts) # Default uncertainty to 0 if not provided
            else:
                raise ValueError(f"Unexpected number of items in DataLoader batch: {len(batch)}")

            x = images.to(device)
            # y = true_labels.to(device) # We don't need labels on device here

            if xtoc_model.arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y_SIGMOID]:
                 # Assuming the first output is always concept logits/predictions
                pred_concepts_output = xtoc_model(x)
                if isinstance(pred_concepts_output, tuple):
                    pred_concepts = pred_concepts_output[0]
                else:
                    pred_concepts = pred_concepts_output # Handle cases where model returns only one tensor
            else:
                raise ValueError(f'Model architecture {xtoc_model.arch} not supported for getting concept predictions.')

            true_concepts_list.append(true_concepts.cpu())
            concept_uncertainty_list.append(concept_uncertainty.cpu())
            pred_concepts_list.append(pred_concepts.cpu())
            true_labels_list.append(true_labels.cpu()) 

    true_concepts = torch.cat(true_concepts_list, dim=0)
    concept_uncertainty = torch.cat(concept_uncertainty_list, dim=0)
    pred_concepts = torch.cat(pred_concepts_list, dim=0)
    true_labels = torch.cat(true_labels_list, dim=0)
    return true_concepts, concept_uncertainty, pred_concepts, true_labels


def get_intervened_concepts(
        true_concepts: torch.Tensor,          # (N, C) on CPU
        concept_uncertainty: torch.Tensor,    # (N, C) on CPU
        pred_concepts: torch.Tensor,          # (N, C) on CPU (Logits)
        intervention_mask: torch.Tensor,      # (N, C) boolean on CPU
        intervention_format: enum_utils.InterventionFormat,
        logits_min: float,
        logits_max: float,
        probs_min: float,
        probs_max: float,
        device: torch.device,
        include_uncertain: bool = False,
    ) -> torch.Tensor: # Returns intervened concepts on DEVICE
    """Performs intervention on concepts for pre-computed tensors.

    Args:
        true_concepts: tensor of true concepts (batch_size, n_concepts) - CPU
        concept_uncertainty: tensor of concept uncertainty flags (batch_size, n_concepts) - CPU
        pred_concepts: tensor of predicted concept logits (batch_size, n_concepts) - CPU
        intervention_mask: boolean tensor of intervention mask (batch_size, n_concepts) - CPU
        intervention_format: Format for intervened values.
        logits_min/max: Precomputed min/max logits for LOGITS format.
        probs_min/max: Precomputed min/max probs for PROBS format.
        device: Target device for the output tensor.
        include_uncertain: Whether to intervene on concepts marked uncertain.

    Returns:
        torch.Tensor: intervened concepts tensor (batch_size, n_concepts) on the specified device.
    """
    # Move necessary tensors to device
    intervention_mask_dev = intervention_mask.bool().to(device)
    true_concepts_float_dev = true_concepts.float().to(device)
    concept_uncertainty_dev = concept_uncertainty.float().to(device) # Ensure float for comparison/masking
    pred_concepts_dev = pred_concepts.to(device)

    # Create the mask of concepts we *actually* intervene on
    # Start with the requested intervention mask
    effective_intervention_mask = intervention_mask_dev

    if not include_uncertain:
        # Find where uncertainty is 1.0 and intervention is requested
        uncertain_and_intervene = (concept_uncertainty_dev == 1.0) & effective_intervention_mask
        # Set intervention mask to False for these uncertain concepts
        effective_intervention_mask = torch.where(uncertain_and_intervene,
                                                  torch.zeros_like(effective_intervention_mask),
                                                  effective_intervention_mask)

    effective_intervention_mask_float = effective_intervention_mask.float()

    # Determine the values to use for intervention based on format
    if intervention_format is enum_utils.InterventionFormat.LOGITS:
        # Use pre-calculated min/max logits
        intervened_val = torch.where(true_concepts_float_dev == 1.0,
                                     torch.full_like(true_concepts_float_dev, logits_max),
                                     torch.full_like(true_concepts_float_dev, logits_min))
        pred_val = pred_concepts_dev # Keep predictions as logits
    elif intervention_format is enum_utils.InterventionFormat.PROBS:
        # Use pre-calculated min/max probs
        intervened_val = torch.where(true_concepts_float_dev == 1.0,
                                     torch.full_like(true_concepts_float_dev, probs_max),
                                     torch.full_like(true_concepts_float_dev, probs_min))
        pred_val = torch.sigmoid(pred_concepts_dev) # Convert predictions to probs
    elif intervention_format is enum_utils.InterventionFormat.BINARY:
        intervened_val = true_concepts_float_dev # Use true binary values
        # Decide how to represent non-intervened concepts.
        pred_val = torch.sigmoid(pred_concepts_dev) # Convert predictions to probs
    else:
        raise ValueError(f'Intervention format {intervention_format} not supported.')

    # Apply intervention: replace predicted values with intervened values where the mask is True
    intervened_concepts = (intervened_val * effective_intervention_mask_float +
                           pred_val * (1.0 - effective_intervention_mask_float))

    return intervened_concepts # Shape (N, C) on device


def evaluate_intervention_step(
        ctoy_model: nn.Module,
        intervened_concepts: torch.Tensor, # Already on device
        true_labels: torch.Tensor,         # On CPU initially
        metrics: Dict[str, torchmetrics.Metric], # On Device
        loss_fn: nn.Module,                # On Device
        batch_size: int,
        device: torch.device) -> Dict[str, float]:
    """Evaluate CtoY model on intervened concepts for one step."""
    ctoy_model.eval()

    # Move true_labels to device once for the whole evaluation step
    true_labels_dev = true_labels.to(device)

    train_util.reset_metrics(metrics)
    total_loss = 0.0
    n_samples = intervened_concepts.shape[0]

    # Create temporary dataloader for batching the evaluation
    eval_dataset = TensorDataset(intervened_concepts, true_labels_dev)
    # Note: num_workers > 0 might cause issues if TensorDataset tensors are already on CUDA.
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch_idx, (concepts_batch, labels_batch) in enumerate(eval_loader):
            predictions_batch_output = ctoy_model(concepts_batch)
            if isinstance(predictions_batch_output, tuple):
                predictions_batch = predictions_batch_output[0] # Logits, Shape (bs, n_classes_y) or (bs, 1)
            else:
                predictions_batch = predictions_batch_output

            # Calculate loss using temporary variables for shape adjustments
            loss = None
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                # Loss expects (N,) logits and (N,) targets
                preds_for_loss = predictions_batch.squeeze(-1) if predictions_batch.ndim == 2 and predictions_batch.shape[1] == 1 else predictions_batch
                labels_for_loss = labels_batch.float().squeeze(-1) if labels_batch.ndim == 2 and labels_batch.shape[1] == 1 else labels_batch.float()
                # Ensure preds_for_loss and labels_for_loss have the same shape (N,)
                if preds_for_loss.shape != labels_for_loss.shape:
                    # This case might happen if one is (N,) and other is (N, 1) initially and only one got squeezed
                    # Add explicit check or ensure model output/label format is consistent (e.g., both (N,1))
                    logging.warning(f"Shape mismatch for BCE loss calculation: preds {preds_for_loss.shape}, labels {labels_for_loss.shape}")
                    # Attempt to force shapes - adjust based on expected input format
                    if preds_for_loss.ndim == 1 and labels_for_loss.ndim == 2: labels_for_loss = labels_for_loss.squeeze(-1)
                    elif preds_for_loss.ndim == 2 and labels_for_loss.ndim == 1: preds_for_loss = preds_for_loss.squeeze(-1)

                loss = loss_fn(preds_for_loss, labels_for_loss)

            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                # Loss expects (N, C) logits and (N,) targets
                labels_for_loss = labels_batch.long().squeeze(-1) if labels_batch.ndim == 2 and labels_batch.shape[1] == 1 else labels_batch.long()
                loss = loss_fn(predictions_batch, labels_for_loss) # Assuming predictions_batch is (N, C)
            else:
                # Fallback: Use original shapes, might need adjustment per loss type
                labels_batch_target = labels_batch # Or labels_batch.float(), .long() etc.
                loss = loss_fn(predictions_batch, labels_batch_target)

            # Ensure loss is a scalar before item()
            if loss is not None and loss.numel() > 1:
                loss = loss.mean()
            if loss is not None: # Check if loss was calculated
                total_loss += loss.item() * concepts_batch.size(0)

            # Update metrics (only class metrics) - PASS ORIGINAL TENSORS
            class_metrics = {k: v for k, v in metrics.items() if k.startswith('class_')}
            # Pass original predictions_batch (e.g., shape (bs, 1)) and labels_batch (e.g., shape (bs, 1))
            custom_update_metrics(metrics=class_metrics,
                      preds_tuple=(predictions_batch,),
                      targets_tuple=(labels_batch,),
                      n_classes=ctoy_model.n_classes) # Pass n_classes


    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    # Compute final metrics after iterating through all batches
    eval_metrics = train_util.compute_metrics(metrics)
    eval_metrics['loss'] = avg_loss

    # Filter results to only include class metrics and loss
    final_results = {k: v for k, v in eval_metrics.items() if k.startswith('class_') or k == 'loss'}
    return final_results


# --- Intervention Evaluation Function ---
def evaluate_intervention(
    xtoc_model: nn.Module,
    ctoy_model: nn.Module,
    ds_val: DataLoader,
    ctoy_val_metrics: Dict[str, torchmetrics.Metric],
    policy_type: PolicyType,
    policy_args: Dict[str, Any],
    loss_fn: nn.Module,
    n_steps: int,
    device: torch.device,
    batch_size: int = 32,
    include_uncertain: bool = False,
    intervention_format: enum_utils.InterventionFormat = enum_utils.InterventionFormat.LOGITS,
    )-> Tuple[List[Dict[str, float]], List[Optional[List[str]]]]:
    """
    Performs up to n_steps of intervention on the validation set, sample by sample,
    and evaluates the model's class metrics after each step.

    policy_args keys:
        - concept_names: Sequence of strings mapping concept indices to names.
        - budget: Intervention budget for each step (number of groups/concepts).

        If policy_type == PolicyType.RANDOM_CONCEPTS:
            - info_single_concepts: Dictionary with information about single concepts.
        If policy_type == PolicyType.RANDOM_GROUPS:
            - info_structured_costs: Dictionary with information about concept groups.
        else:
            - concept_metric: enum_utils.Metric for concept uncertainty.
            - label_metric: enum_utils.Metric for label-based concept importance.
            - label_metric_weight: Weight for label importance metric relative to concept uncertainty.
            If policy_type == PolicyType.OPTIMIZED or PolicyType.GREEDY_GROUPS:
            - info_structured_costs: Dictionary with information about concept groups.
            If policy_type == PolicyType.GREEDY_CONCEPTS:
            - info_single_concepts: Dictionary with information about single concepts.
    """
    # check correct structure of policy_args
    required_keys = (
    {'concept_names', 'budget', 'info_single_concepts'} if policy_type == PolicyType.RANDOM_CONCEPTS else
    {'concept_names', 'budget', 'info_structured_costs'} if policy_type == PolicyType.RANDOM_GROUPS else
    {'concept_names', 'budget', 'info_single_concepts', 'concept_metric', 'label_metric', 'label_metric_weight'} if policy_type == PolicyType.GREEDY_CONCEPTS else # Added concept_metric
    {'concept_names', 'budget', 'info_structured_costs', 'concept_metric', 'label_metric', 'label_metric_weight'}) # Added concept_metric

    assert policy_args.keys() >= required_keys, f"Missing required keys for {policy_type}: {required_keys - policy_args.keys()}"

    # unpack policy_args
    concept_names = policy_args['concept_names']
    budget = policy_args['budget']
    info_single_concepts = policy_args.get('info_single_concepts', None)
    info_structured_costs = policy_args.get('info_structured_costs', None)
    concept_metric = policy_args.get('concept_metric', None)
    label_metric = policy_args.get('label_metric', None)
    label_metric_weight = policy_args.get('label_metric_weight', None)

    xtoc_model.eval()
    ctoy_model.eval()
    n_concepts = len(concept_names)
    concept_idx_map = {name: i for i, name in enumerate(concept_names)} # Map name back to index

    # 1. Pre-compute X->C predictions (Result Tensors are on CPU)
    true_concepts, concept_uncertainty, pred_concepts_logits, true_labels = _get_true_predicted_concepts_true_labels(xtoc_model, ds_val, device)
    n_samples = true_concepts.shape[0]
    logging.info(f"Pre-computed X->C predictions for {n_samples} samples.")

    # 2. Pre-compute intervention values (min/max logits/probs) - Use device for quantiles
    pred_concepts_logits_dev = pred_concepts_logits.to(device)
    pred_probs_dev = torch.sigmoid(pred_concepts_logits_dev)
    probs_min = torch.quantile(pred_probs_dev.flatten(), 0.05).item()
    probs_max = torch.quantile(pred_probs_dev.flatten(), 0.95).item()
    del pred_concepts_logits_dev, pred_probs_dev # free gpu memory
    torch.cuda.empty_cache()


    # For LOGITS format, we need robust min/max. Avoid direct quantiles on raw logits.
    # Use min/max of the *sigmoid* probabilities mapped back to logits.
    # This prevents extreme outliers in logits from dominating the range.
    epsilon = 1e-6 # To avoid log(0) or log(1) issues
    safe_probs_min = torch.clamp(torch.tensor(probs_min), epsilon, 1.0 - epsilon)
    safe_probs_max = torch.clamp(torch.tensor(probs_max), epsilon, 1.0 - epsilon)
    logits_min = torch.log(safe_probs_min / (1.0 - safe_probs_min)).item()
    logits_max = torch.log(safe_probs_max / (1.0 - safe_probs_max)).item()
    logging.info(f"Intervention ranges: Logits [{logits_min:.4f}, {logits_max:.4f}], Probs [{probs_min:.4f}, {probs_max:.4f}]")


    # 3. Initialize intervention state
    intervention_metrics_history = []
    concepts_revealed_at_step = []
    intervention_mask = torch.zeros((n_samples, n_concepts), dtype=torch.bool) # False = not intervened, True = intervened. on CPU.

    # --- Intervention Loop ---
    label_metric_fnc = get_metric_fn(label_metric)
    intervened_concepts_tensor_dev = None # Keep track of the tensor on device
    c_uncertainty_scores = None # Initialize uncertainty scores

    for s in range(n_steps + 1): # Evaluate state *before* step 0 and *after* each step 1..n_steps
        logging.info(f"--- Intervention Step {s}/{n_steps} ---")

        # 5.a Evaluate current state
        # Generate intervened concepts for this step on the fly on DEVICE
        intervened_concepts_tensor_dev = get_intervened_concepts(
            true_concepts, concept_uncertainty, pred_concepts_logits,
            intervention_mask, # Pass CPU mask
            intervention_format,
            logits_min, logits_max, probs_min, probs_max,
            device, include_uncertain=include_uncertain
        )

        # Evaluate C->Y on these concepts (pass tensors already on device)
        step_metrics = evaluate_intervention_step(
            ctoy_model, intervened_concepts_tensor_dev, true_labels, # true_labels is on CPU here
            ctoy_val_metrics, loss_fn, batch_size, device
        )
        intervention_metrics_history.append(step_metrics)
        logging.info(f"Step {s} evaluation metrics: {step_metrics}")

        # Store revealed concepts for the *previous* step (i.e., what was revealed to get to state s)
        # For s=0, nothing was revealed yet. For s=1, store what was revealed in the s=0 selection phase.
        if s > 0:
             # concepts_revealed_at_step[s-1] contains the list of names revealed just before this evaluation
             pass # Already appended at the end of the previous iteration
        if s == 0:
             concepts_revealed_at_step.append(None) # Placeholder for step 0

        # Log revealed concepts (handle None for step 0)
        logging.info(f"Step {s} revealed concepts: {concepts_revealed_at_step[s] if s > 0 else 'None'}")


        if torch.all(intervention_mask):
            logging.info("All concepts have been intervened. Ending intervention.")
            intervened_concepts_tensor_dev = get_intervened_concepts(
            true_concepts, concept_uncertainty, pred_concepts_logits,
            intervention_mask, # Pass CPU mask
            intervention_format,
            logits_min, logits_max, probs_min, probs_max,
            device, include_uncertain=include_uncertain
            )

            # Evaluate C->Y on these concepts (pass tensors already on device)
            step_metrics = evaluate_intervention_step(
                ctoy_model, intervened_concepts_tensor_dev, true_labels, # true_labels is on CPU here
                ctoy_val_metrics, loss_fn, batch_size, device
            )
            intervention_metrics_history.append(step_metrics)
            logging.info(f"Last step evaluation metrics: {step_metrics}")
            break


        # --- 5b. If not the last evaluation, select concepts for the *next* intervention step (s+1) ---
        if s < n_steps:
            final_mask_from_policy = None # Initialize
            step_concepts_revealed = None # Initialize

            if policy_type == PolicyType.RANDOM_CONCEPTS:
                final_mask_from_policy, step_concepts_revealed = policies.random_concepts(
                    intervention_mask, info_single_concepts, concept_names, budget, include_uncertain, concept_uncertainty)

            elif policy_type == PolicyType.RANDOM_GROUPS:
                final_mask_from_policy, step_concepts_revealed = policies.random_groups(
                    intervention_mask, info_structured_costs, concept_names, budget, include_uncertain, concept_uncertainty)

            elif policy_type in [PolicyType.GREEDY_CONCEPTS, PolicyType.GREEDY_GROUPS, PolicyType.OPTIMIZED]:

                # Pre-compute concept uncertainty scores (once, as they don't change)
                # Compute on CPU to avoid large tensor on GPU for potentially many concepts
                if s == 0: # Compute only before the first selection step
                     concept_metric_fnc = get_metric_fn(concept_metric)
                     # Ensure metric function takes logits directly
                     c_uncertainty_scores = concept_metric_fnc(pred_concepts_logits) # Shape (N, C) on CPU
                     logging.info("Calculated concept uncertainty scores.")

                # Compute label-based importance scores (depends on current state, needs recomputation)
                # Compute on CPU to avoid multiple large tensors on GPU during iteration
                concept_importance_scores = torch.zeros_like(pred_concepts_logits, device='cpu')
                if label_metric_weight > 1e-6:
                    logging.info(f"Step {s}: Calculating label importance scores...")
                    # determine intervention values for 0 and 1
                    if intervention_format == enum_utils.InterventionFormat.LOGITS: value_0, value_1 = logits_min, logits_max
                    elif intervention_format == enum_utils.InterventionFormat.PROBS: value_0, value_1 = probs_min, probs_max
                    elif intervention_format == enum_utils.InterventionFormat.BINARY: value_0, value_1 = 0.0, 1.0
                    else: raise ValueError(f"Unsupported intervention format: {intervention_format}")

                    # --- Compute concept importances in batches ---
                    # Create a temporary DataLoader for batching this calculation
                    temp_dataset = TensorDataset(
                         intervened_concepts_tensor_dev, # Current concepts (on device)
                         pred_concepts_logits # Original X->C logits (on CPU, moved in loop)
                    )
                    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # num_workers=0 for CUDA tensors

                    batch_start_idx = 0
                    with torch.no_grad():
                        for current_concepts_batch_dev, pred_concepts_logits_batch_cpu in temp_loader:
                            batch_size_actual = current_concepts_batch_dev.shape[0]
                            batch_indices = slice(batch_start_idx, batch_start_idx + batch_size_actual)
                            pred_concepts_logits_batch_dev = pred_concepts_logits_batch_cpu.to(device)

                            # Get current label prediction logits for this batch
                            current_label_logits_batch_dev = ctoy_model(current_concepts_batch_dev)[0] # Get logits directly
                            # Reshape if necessary (e.g., binary case might output (N,) )
                            if ctoy_model.n_classes == 1 and current_label_logits_batch_dev.ndim == 1:
                                current_label_logits_batch_dev = current_label_logits_batch_dev.unsqueeze(1) # Ensure (N, 1)

                            # Initialize tensors for label predictions when intervening on each concept *individually*
                            bs, nC = current_concepts_batch_dev.shape
                            n_classes_y = ctoy_model.n_classes
                            pred_labels_0_logits_batch_dev = torch.zeros((batch_size_actual, nC, n_classes_y), device=device, dtype=current_concepts_batch_dev.dtype)
                            pred_labels_1_logits_batch_dev = torch.zeros((batch_size_actual, nC, n_classes_y), device=device, dtype=current_concepts_batch_dev.dtype)

                            # Iterate through each concept *column* to simulate intervention
                            for j in range(nC):
                                 concepts_0_dev = current_concepts_batch_dev.clone()
                                 concepts_1_dev = current_concepts_batch_dev.clone()
                                 concepts_0_dev[:, j], concepts_1_dev[:, j] = value_0, value_1
                                 logits_0_dev = ctoy_model(concepts_0_dev)[0] # Assume first output is logits
                                 logits_1_dev = ctoy_model(concepts_1_dev)[0]

                                 # Ensure shape consistency for binary case (N, 1)
                                 if n_classes_y == 1:
                                     if logits_0_dev.ndim == 1: logits_0_dev = logits_0_dev.unsqueeze(1)
                                     if logits_1_dev.ndim == 1: logits_1_dev = logits_1_dev.unsqueeze(1)

                                 pred_labels_0_logits_batch_dev[:, j, :] = logits_0_dev
                                 pred_labels_1_logits_batch_dev[:, j, :] = logits_1_dev

                            # calculate importance score for this batch
                            if n_classes_y > 1:
                                raise NotImplementedError("Multi-class label metric calculation not implemented yet.")
                            else:
                                batch_importance = label_metric_fnc(
                                    pred_concepts_logits_batch_dev, # Orig X->C preds
                                    current_label_logits_batch_dev, # Current C->Y preds
                                    pred_labels_0_logits_batch_dev, # C->Y preds if Cj=0
                                    pred_labels_1_logits_batch_dev  # C->Y preds if Cj=1
                                )

                            # Store batch results back to CPU tensor
                            concept_importance_scores[batch_indices] = batch_importance.cpu()
                            batch_start_idx += batch_size_actual # Move start index for next slice

                    # --- end of batch loop ---

                    del temp_loader, temp_dataset, current_concepts_batch_dev, pred_concepts_logits_batch_dev, current_label_logits_batch_dev, pred_labels_0_logits_batch_dev, pred_labels_1_logits_batch_dev, batch_importance
                    # Free memory from loop variables if they exist
                    if 'concepts_0_dev' in locals(): del concepts_0_dev # Use locals() check for safety
                    if 'concepts_1_dev' in locals(): del concepts_1_dev
                    if 'logits_0_dev' in locals(): del logits_0_dev
                    if 'logits_1_dev' in locals(): del logits_1_dev
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                # --- end if label_metric_weight > 1e-6 ---
                logging.info(f"Step {s}: Combining scores...")
                # Ensure uncertainty scores are ready before combining
                if c_uncertainty_scores is None:
                     raise RuntimeError("Concept uncertainty scores were not computed.")

                c_uncertainty_scores = torch.nan_to_num(c_uncertainty_scores, nan=0.0, posinf=0.0, neginf=0.0)
                concept_importance_scores = torch.nan_to_num(concept_importance_scores, nan=0.0, posinf=0.0, neginf=0.0)
                combined_scores = (1.0 - label_metric_weight) * c_uncertainty_scores + label_metric_weight * concept_importance_scores

                # --- Call the appropriate policy function ---
                logging.info(f"Step {s}: Running policy '{policy_type.value}'...")
                if policy_type == PolicyType.GREEDY_CONCEPTS:
                    final_mask_from_policy, step_concepts_revealed = policies.greedy_concepts(
                        global_intervention_mask=intervention_mask,
                        concept_info=info_single_concepts,
                        concept_values=combined_scores,
                        concept_names=concept_names,
                        budget=budget,
                        include_uncertain=include_uncertain,
                        concept_uncertainty=concept_uncertainty
                    )
                elif policy_type == PolicyType.GREEDY_GROUPS:
                    final_mask_from_policy, step_concepts_revealed = policies.greedy_groups(
                        global_intervention_mask=intervention_mask,
                        structured_costs=info_structured_costs,
                        concept_values=combined_scores,
                        concept_names=concept_names,
                        budget=budget,
                        include_uncertain=include_uncertain,
                        concept_uncertainty=concept_uncertainty
                    )
                elif policy_type == PolicyType.OPTIMIZED:
                    final_mask_from_policy, step_concepts_revealed = policies.optimized(
                        global_intervention_mask=intervention_mask,
                        structured_costs=info_structured_costs,
                        concept_values=combined_scores,
                        concept_names=concept_names,
                        budget=budget,
                        include_uncertain=include_uncertain,
                        concept_uncertainty=concept_uncertainty
                    )
            else:
                raise ValueError(f"Policy type {policy_type} not supported.")

            # --- End of policy selection ---

            # update global state after policy selection
            if final_mask_from_policy is not None:
                intervention_mask = final_mask_from_policy
            concepts_revealed_at_step.append(step_concepts_revealed) # Always append, even if None or empty

            # if for all the samples and concepts the intervention mask is True, break
            if torch.all(intervention_mask):
                logging.info("All concepts have been intervened. Ending intervention.")
                intervened_concepts_tensor_dev = get_intervened_concepts(
                true_concepts, concept_uncertainty, pred_concepts_logits,
                intervention_mask, # Pass CPU mask
                intervention_format,
                logits_min, logits_max, probs_min, probs_max,
                device, include_uncertain=include_uncertain
                )

                # Evaluate C->Y on these concepts (pass tensors already on device)
                step_metrics = evaluate_intervention_step(
                    ctoy_model, intervened_concepts_tensor_dev, true_labels, # true_labels is on CPU here
                    ctoy_val_metrics, loss_fn, batch_size, device
                )
                intervention_metrics_history.append(step_metrics)
                logging.info(f"Last step evaluation metrics: {step_metrics}")
                break

        # --- End of step s < n_steps block ---

        # Free up GPU memory from the evaluation step if the tensor exists
        if intervened_concepts_tensor_dev is not None:
            del intervened_concepts_tensor_dev
            intervened_concepts_tensor_dev = None
            torch.cuda.empty_cache()

    # --- End of Intervention Loop ---

    # --- 6. Return results ---
    # intervention_metrics_history: List of dicts (metrics per step)
    # concepts_revealed_at_step: List of lists (unique concept names revealed *at* each step s)
    return intervention_metrics_history, concepts_revealed_at_step



# --- Main Execution Block ---
def main(argv: Sequence[str]):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    config = FLAGS

    # --- Load Data: support only CheXpert ---
    logging.info(f"Loading dataset: {config.dataset.name}")
    if config.dataset == enum_utils.Dataset.CHEXPERT:
        dataset_module = chexpert_dataset
    else:
        raise ValueError('Dataset not supported.')
    

    for seed in config.training_seeds:
        start_time = time.time()

        # Load validation set for intervention evaluation
        if config.merge_train_and_val:
            raise ValueError('merge_train_and_val is not supported for intervention evaluation.')
        _, ds_val, _ = dataset_module.load_dataset(config.batch_size, config.merge_train_and_val, config.num_workers) # already has shuffle=False

        # --- Dataset/Model Config ---
        n_classes = dataset_module.Config.n_classes
        n_concepts = dataset_module.Config.n_concepts
        non_linear_ctoy = config.non_linear_ctoy
        concept_names = dataset_module.load_concept_names() # list
        info_structured_costs, _ = dataset_module.load_hardcoded_detailed_costs()
        single_concept_info = dataset_module.get_min_cost_per_concept()

        # --- Load Model ---
        # Determine path based on flags or defaults
        seed_in_path = seed if config.training_seeds is not None else 'None'
        print(f'Using seed in path: {seed_in_path}')
        if config.bottleneck_type == enum_utils.BottleneckType.INDEPENDENT:
            btype = 'independent'
        elif config.bottleneck_type == enum_utils.BottleneckType.JOINT:
            btype = 'joint'
        elif config.bottleneck_type == enum_utils.BottleneckType.JOINT_SIGMOID:
            btype = 'joint_sigmoid'
        else:
            btype = None

       
        if seed < 0: 
            config_seed = 'None'
        else: 
            config_seed = seed

        # tutto da sistemare, se è independent, devi ri-definirlo
        # path_exp_dir = os.path.join(
        #     config.checkpoint_dir, config.dataset.value, btype, config.arch.value, f'seed-{config_seed}',
        #     f'{config.optimizer}_lr-{config.lr}_wd-{config.wd}'
        # )
        path_exp_dir = os.path.join(config.checkpoint_dir, config.dataset.value, btype)
        # here you should put the arch.value depending on the btype you are using
        suffix = os.path.join(f'seed-{config_seed}', f'{config.optimizer}_lr-{config.lr}_wd-{config.wd}', 'checkpoint_best_class_auroc.pth.tar')
            
        logging.info(f"Loading models using bottleneck type: {config.bottleneck_type.name}")

        xtoc_model = None
        ctoy_model = None

        if config.bottleneck_type == enum_utils.BottleneckType.INDEPENDENT:
            # fill the path 
            # Assuming concept checkpoint uses 'concept' in name, adjust if needed
            xtoc_suffix = suffix.replace('class_auroc', 'concept_auroc') # Example: adjust based on actual naming
            xtoc_path = path_exp_dir + '/XtoC/' + xtoc_suffix
            ctoy_path = path_exp_dir + '/CtoY/' + suffix

            logging.info(f"Loading XtoC model from: {xtoc_path}")
            xtoc_model = network.InteractiveBottleneckModel(arch=enum_utils.Arch.X_TO_C, n_concepts=n_concepts, n_classes=n_classes).to(device)
            # Handle potential file not found for specific concept metric checkpoint
            try:
                _, _ = utils.load_checkpoint(xtoc_path, xtoc_model)
            except FileNotFoundError:
                print(suffix)
                print(xtoc_suffix)
                raise FileNotFoundError(f"Checkpoint not found: {xtoc_path}. Please check the path or file name.")
                logging.warning(f"XtoC checkpoint '{xtoc_path}' not found. Trying 'last' checkpoint.")
                xtoc_path = path_exp_dir + '/XtoC' + '/sgd_lr-0.001_wd-0.0/checkpoint_last.pth.tar' # Fallback path
                utils.load_checkpoint(xtoc_path, xtoc_model)

            logging.info(f"Loading CtoY model from: {ctoy_path}")
            ctoy_model = network.InteractiveBottleneckModel(arch=enum_utils.Arch.C_TO_Y, n_concepts=n_concepts, n_classes=n_classes, non_linear_ctoy=non_linear_ctoy).to(device)
            utils.load_checkpoint(ctoy_path, ctoy_model) # Pass device

            xtoc_model = xtoc_model.to(device)
            ctoy_model = ctoy_model.to(device)

        elif config.bottleneck_type in [enum_utils.BottleneckType.JOINT, enum_utils.BottleneckType.JOINT_SIGMOID]:
            # Determine architecture and path for the joint model
            if config.bottleneck_type == enum_utils.BottleneckType.JOINT_SIGMOID:
                arch = enum_utils.Arch.X_TO_C_TO_Y_SIGMOID
                model_dir = '/XtoCtoY_sigmoid' # Assuming directory name convention
            else: # JOINT
                arch = enum_utils.Arch.X_TO_C_TO_Y
                model_dir = '/XtoCtoY' # Assuming directory name convention

            # Construct the full path using the hardcoded base and suffix from the original script
            path = path_exp_dir + model_dir + suffix
            logging.info(f"Loading joint model from: {path}")

            # Load the full joint model first
            full_model = network.InteractiveBottleneckModel(arch=arch, n_concepts=n_concepts, n_classes=n_classes, non_linear_ctoy=non_linear_ctoy).to(device)
            _, _ = utils.load_checkpoint(path, full_model)
            full_state_dict = full_model.state_dict()
            del full_model # Free memory
            torch.cuda.empty_cache()

            xtoc_model = network.InteractiveBottleneckModel(arch=enum_utils.Arch.X_TO_C, n_concepts=n_concepts, n_classes=n_classes).to(device)
            ctoy_model = network.InteractiveBottleneckModel(arch=enum_utils.Arch.C_TO_Y, n_concepts=n_concepts, n_classes=n_classes, non_linear_ctoy=non_linear_ctoy).to(device)

            xtoc_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith('base_model.') or k.startswith('gap.') or k.startswith('concept_layer.')}
            ctoy_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith('ctoy_module.')}


            xtoc_model.load_state_dict(xtoc_state_dict, strict=True)
            ctoy_model.load_state_dict(ctoy_state_dict, strict=True)

            xtoc_model.to(device)
            ctoy_model.to(device)

        else:
            raise ValueError(f'Bottleneck type {config.bottleneck_type} not supported.')

        # Ensure models are in eval mode
        if xtoc_model: xtoc_model.eval()
        if ctoy_model: ctoy_model.eval()

        # --- Prepare Evaluation ---
        # Loss Function
        if n_classes == 1:
            # For binary classification with logits output
            loss_fn = nn.BCEWithLogitsLoss().to(device)
        else:
            # For multi-class classification with logits output
            loss_fn = nn.CrossEntropyLoss().to(device)

        # Metrics for CtoY evaluation (move metrics to device)
        _, ctoy_val_metrics = train_util.get_metrics(enum_utils.Arch.C_TO_Y, n_concepts, n_classes, device)
        logging.info("Initialized CtoY evaluation metrics.")


        # --- Run Intervention Evaluation ---
        policy_args = {
            'concept_names': concept_names,
            'budget': config.budget,
            'concept_metric': config.concept_metric, 
            'label_metric': config.label_metric,
            'label_metric_weight': config.label_metric_weight,
            'info_single_concepts': single_concept_info,
            'info_structured_costs': info_structured_costs,
        }
        intervention_metrics_history, concepts_revealed_history = evaluate_intervention(
            xtoc_model=xtoc_model,
            ctoy_model=ctoy_model,
            ds_val=ds_val,
            ctoy_val_metrics=ctoy_val_metrics,
            policy_type=PolicyType(config.policy_type), 
            policy_args=policy_args,
            loss_fn=loss_fn,
            n_steps=config.n_intervention_steps,
            device=device,
            batch_size=config.batch_size,
            intervention_format=config.intervention_format,
            include_uncertain=config.include_uncertain_in_intervention,
        )
        logging.info("Intervention evaluation finished.")

        # --- Save Results ---
        # Convert flag values to serializable types
        serializable_config = {}
        flags_dict = config.flag_values_dict() # Get all flags
        for name, value in flags_dict.items():
            if isinstance(value, (enum_utils.Arch, enum_utils.Dataset, enum_utils.BottleneckType, enum_utils.Metric, enum_utils.InterventionFormat, PolicyType)):
                serializable_config[name] = value.name # Save enum name
            elif isinstance(value, (int, float, bool, str, list, dict, type(None))):
                serializable_config[name] = value
            else:
                serializable_config[name] = repr(value) # Fallback for other types


        results = {
            'config': serializable_config, # Save serializable flags
            'metrics_history': intervention_metrics_history, # List of dicts (step -> metric -> value)
            'concepts_revealed_history': concepts_revealed_history, # List of lists (step -> unique concept names revealed)
            'concept_names': concept_names, # List of concept names
            'n_samples_evaluated': len(ds_val.dataset),
            'evaluation_time_seconds': time.time() - start_time,
        }

        results_path = config.results_file
        datetime_short = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = path_exp_dir + "/" + config.policy_type + "_B" + str(config.budget) + "_" + datetime_short +  ".json" #modifica
        results_dir = os.path.dirname(results_path)
        if results_dir: # Handle case where path is just a filename
            os.makedirs(results_dir, exist_ok=True)

        logging.info(f"Attempting to save results to: {results_path}")
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            logging.info(f"Results successfully saved to {results_path}")
        except TypeError as e:
            logging.error(f"Error saving results to JSON: {e}. Check data types in results dict.", exc_info=True)
            # Fallback: print basic info to console
            print("\n--- Intervention Results (JSON serialization failed) ---")
            print("Config:", serializable_config)
            print("\nMetrics History:")
            for i, metrics in enumerate(intervention_metrics_history):
                print(f"  Step {i}: {metrics}")
            print("\nConcepts Revealed History:")
            for i, concepts in enumerate(concepts_revealed_history):
                print(f"  Step {i}: {concepts}")
            print("-------------------------------------------------------\n")

        total_time = time.time() - start_time
        logging.info(f"Total execution time: {total_time:.2f} seconds.")

    
if __name__ == '__main__':
    app.run(main)