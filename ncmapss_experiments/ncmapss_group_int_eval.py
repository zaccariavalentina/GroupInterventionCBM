import os
import sys 
import datetime
import enum
import time
import logging
from typing import Sequence, Dict, Any, Tuple, Optional, Callable, List
import json
import warnings
from functools import partial
import random

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from absl import flags
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchmetrics
import numpy as np
from pytorch_lightning import seed_everything
from torch.distributions import Bernoulli

from data.ncmapss import NCMAPSSDataset
from data.cmapss import CMAPSSDataset
from cem.models.cem_regression import ConceptEmbeddingModel as CEM_Regression
from models.cem import latent_cnn_code_generator_model, latent_mlp_code_generator_model
from GroupInterventionCBM import policies as baseline_policies 


warnings.filterwarnings("ignore")

# --- Configuration Flags ---
_ARCHITECTURE = flags.DEFINE_enum('architecture', default="ConceptEmbeddingModel", enum_values=["ConceptEmbeddingModel", "ConceptBottleneckModel"], help='Architecture type to load (must match training).')
_EMB_SIZE = flags.DEFINE_integer('emb_size', default=16, help='Embedding size per concept (for CEM).')
_C_EXTRACTOR_ARCH = flags.DEFINE_string('c_extractor_arch', default="cnn", help='Feature extractor backbone architecture used during training (cnn or mlp).')
_DATASET = flags.DEFINE_enum('dataset', default="N-CMAPSS", enum_values=["CMAPSS", "N-CMAPSS"], help='Dataset to use for evaluation.')

# chenge with your dataset path
_DATA_PATH = flags.DEFINE_string('data_path', default='/mnt/disk1/valentina_zaccaria/ncmapss/', help='Path to dataset files.')

_OUTPUT_DIR = flags.DEFINE_string('output_dir', default='./results/cmapss_intervention/', help='Directory to load models and save results.')
_CHECKPOINT_SUFFIX = flags.DEFINE_string('checkpoint_suffix', default='.pt', help='Suffix for checkpoint files (e.g., .pt or .ckpt).')
_SPLIT = flags.DEFINE_integer('split', default=0, help='Data split/fold index (if applicable).')
_N_DS = flags.DEFINE_list('n_ds', default=["01-005", "04", "05", "07"], help='List of dataset identifiers (e.g., ["01", "02"] for N-CMAPSS).')
_TEST_UNITS = flags.DEFINE_list('test_units', default=[7, 8, 9, 10], help='Specific units to test (optional). Use empty list for all units.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', default=256, help='Batch Size for evaluation steps.')
_NUM_WORKERS = flags.DEFINE_integer('num_workers', default=4, help='Number of DataLoader workers.')
_WINDOW_SIZE = flags.DEFINE_integer('window_size', default=50, help='Input window size.')
_STRIDE = flags.DEFINE_integer('stride', default=1, help='Data loading stride.')
_SCALING = flags.DEFINE_string('scaling', default='legacy', help='Data scaling method used during training.')
_DOWNSAMPLE = flags.DEFINE_integer('downsample', default=10, help='Downsampling rate for test data.')
_RUL_TYPE = flags.DEFINE_enum('rul_type', default='flat', enum_values=['linear', 'flat'], help='Type of RUL calculation used during training.')
_CONCEPTS = flags.DEFINE_list('concepts', default=["Fan-E", "Fan-F", "LPC-E", "LPC-F", "HPC-E", "HPC-F", "LPT-E", "LPT-F", "HPT-E", "HPT-F"], help='List of concept names used.')
_BINARY_CONCEPTS = flags.DEFINE_bool('binary_concepts', default=True, help='Whether concepts were treated as binary.')
_COMBINED_CONCEPTS = flags.DEFINE_bool('combined_concepts', default=False, help='Whether combined concepts were used (ensure False).')

# policy type and metrics 
_POLICY_TYPE = flags.DEFINE_enum('policy_type', default='optimized', enum_values=['random_concepts', 'random_groups', 'greedy_concepts', 'greedy_groups', 'optimized'], help='Type of policy for concept/group selection.')
_N_INTERVENTION_STEPS = flags.DEFINE_integer('n_intervention_steps', default=1, help='Number of intervention steps.')
_INTERVENTION_BUDGET = flags.DEFINE_float('budget', default=3.5, help='Budget for intervention selection (number for count-based, cost for cost-based). Assume count-based for now.')
_CONCEPT_METRIC = flags.DEFINE_enum('concept_metric', default="CONCEPT_ENTROPY", enum_values=["CONCEPT_ENTROPY", "CONCEPT_CONFIDENCE"], help='Metric for concept uncertainty (for greedy policies).')
_LABEL_METRIC = flags.DEFINE_enum('label_metric', default="LABEL_ERROR_DECREASE", enum_values=["LABEL_ERROR_DECREASE", "LABEL_ERROR_CHANGE"], help='Metric for label-based importance (for greedy policies).')
_LABEL_METRIC_WEIGHT = flags.DEFINE_float('label_metric_weight', default=0, help='Weighting for label importance vs concept uncertainty.')
_INTERVENTION_FORMAT = flags.DEFINE_enum('intervention_format', default="BINARY", enum_values=["BINARY", "PROBS"], help='Format for intervened concepts (BINARY uses 0/1, PROBS uses min/max probabilities).')
_INCLUDE_UNCERTAIN_IN_INTERVENTION = flags.DEFINE_bool('include_uncertain_in_intervention', default=True, help='Whether to allow intervention on concepts marked as uncertain (if applicable).')


_RESULTS_FILE_SUFFIX = flags.DEFINE_string('results_file_suffix', default='intervention_results.json', help='Suffix for the results JSON file.')
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string('checkpoint', default=None, help='Direct path to the specific model checkpoint file (.ckpt or .pt) to load.')
_SEED = flags.DEFINE_integer('seed', default=0, help='seed.')

# subsample size of the test set for evaluation
_EVAL_SUBSAMPLE_SIZE = flags.DEFINE_integer('eval_subsample_size', default=20000, help='Number of samples (windows) to randomly subsample for evaluation. If None or 0, use the full dataset.')


FLAGS = flags.FLAGS

# --- Enum Definition ---
class PolicyType(str, enum.Enum):
    RANDOM_CONCEPTS: str = 'random_concepts'
    RANDOM_GROUPS: str = 'random_groups'
    GREEDY_CONCEPTS: str = 'greedy_concepts'
    GREEDY_GROUPS: str = 'greedy_groups'
    OPTIMIZED: str = 'optimized'

# --- Policy Metrics ---
class CustomPolicyMetrics:
    @staticmethod
    def concept_entropy(pred_concepts_logits: torch.Tensor) -> torch.Tensor:
        clamped_logits = torch.clamp(pred_concepts_logits, min=-15, max=15)
        try: entropy = Bernoulli(logits=clamped_logits).entropy()
        except ValueError as e:
             logging.error(f"Error calculating Bernoulli entropy: {e}. Logits range: [{clamped_logits.min()}, {clamped_logits.max()}]. Returning zeros.")
             entropy = torch.zeros_like(clamped_logits)
        return torch.nan_to_num(entropy, nan=0.0)

    @staticmethod
    def concept_confidence(pred_concepts_logits: torch.Tensor) -> torch.Tensor:
        concept_probs = torch.sigmoid(pred_concepts_logits)
        return torch.max(concept_probs, 1.0 - concept_probs)

    @staticmethod
    def _label_metric_base(concept_logits: torch.Tensor,
                           current_label_pred: torch.Tensor,
                           label_pred_if_0: torch.Tensor,
                           label_pred_if_1: torch.Tensor,
                           metric_type: str,
                           signed: bool) -> torch.Tensor:
        if current_label_pred.ndim == 1: current_label_pred = current_label_pred.unsqueeze(-1)
        if current_label_pred.ndim != 2 or current_label_pred.shape[1] != 1:
             logging.error(f"Unexpected shape for current_label_pred: {current_label_pred.shape}. Expected [N, 1]. Attempting reshape.")
             if current_label_pred.numel() == concept_logits.shape[0]: current_label_pred = current_label_pred.view(-1, 1)
             else: raise ValueError(f"Cannot reshape current_label_pred {current_label_pred.shape}")
        assert label_pred_if_0.shape == label_pred_if_1.shape == concept_logits.shape, "Shape mismatch in label_metric_base"
        assert current_label_pred.shape[0] == concept_logits.shape[0], "Batch size mismatch in label_metric_base"

        concept_probs = torch.sigmoid(concept_logits)
        current_label_pred_expanded = current_label_pred.expand(-1, concept_logits.shape[1])
        diff_1 = label_pred_if_1 - current_label_pred_expanded
        diff_0 = label_pred_if_0 - current_label_pred_expanded

        if metric_type == 'LABEL_ERROR_CHANGE':
            metric = concept_probs * torch.abs(diff_1) + (1.0 - concept_probs) * torch.abs(diff_0)
        elif metric_type == 'LABEL_ERROR_DECREASE':
            metric = concept_probs * (diff_1 ** 2) + (1.0 - concept_probs) * (diff_0 ** 2)
        else: raise ValueError(f"Unsupported metric_type: {metric_type}")
        return torch.nan_to_num(metric, nan=0.0, posinf=1e9, neginf=-1e9)

    @staticmethod
    def label_error_change(concept_logits: torch.Tensor, current_label_pred: torch.Tensor,
                           label_pred_if_0: torch.Tensor, label_pred_if_1: torch.Tensor, signed: bool):
        return CustomPolicyMetrics._label_metric_base(concept_logits, current_label_pred, label_pred_if_0, label_pred_if_1, 'LABEL_ERROR_CHANGE', signed=False)

    @staticmethod
    def label_error_decrease(concept_logits: torch.Tensor, current_label_pred: torch.Tensor,
                             label_pred_if_0: torch.Tensor, label_pred_if_1: torch.Tensor, signed: bool):
        return CustomPolicyMetrics._label_metric_base(concept_logits, current_label_pred, label_pred_if_0, label_pred_if_1, 'LABEL_ERROR_DECREASE', signed=False)


# --- Utility Functions (Unchanged) ---
def get_metric_fn(metric_name: str) -> Callable:
    # ... (implementation as before) ...
    if metric_name == "CONCEPT_ENTROPY": return CustomPolicyMetrics.concept_entropy
    elif metric_name == "CONCEPT_CONFIDENCE": return CustomPolicyMetrics.concept_confidence
    elif metric_name == "LABEL_ERROR_CHANGE": return partial(CustomPolicyMetrics.label_error_change, signed=False)
    elif metric_name == "LABEL_ERROR_DECREASE": return partial(CustomPolicyMetrics.label_error_decrease, signed=False)
    else: raise ValueError(f'Metric {metric_name} not supported.')


def get_cmapss_concept_costs() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    structured_costs = { 'Fan': {'setup_cost': 0.1, 'concepts': {'Fan-E': 1.0, 'Fan-F': 1.0}}, 'LPC': {'setup_cost': 0.1, 'concepts': {'LPC-E': 1.0, 'LPC-F': 1.0}}, 'HPC': {'setup_cost': 0.1, 'concepts': {'HPC-E': 1.0, 'HPC-F': 1.0}}, 'LPT': {'setup_cost': 0.1, 'concepts': {'LPT-E': 1.0, 'LPT-F': 1.0}}, 'HPT': {'setup_cost': 0.1, 'concepts': {'HPT-E': 1.0, 'HPT-F': 1.0}}, }
    single_concept_info = { 'Fan-E': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'Fan'}, 'Fan-F': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'Fan'}, 'LPC-E': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'LPC'}, 'LPC-F': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'LPC'}, 'HPC-E': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'HPC'}, 'HPC-F': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'HPC'}, 'LPT-E': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'LPT'}, 'LPT-F': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'LPT'}, 'HPT-E': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'HPT'}, 'HPT-F': {'marginal_cost': 1.0, 'setup_cost': 0.1, 'group': 'HPT'}, }
    return structured_costs, single_concept_info



# --- NASA Score ---
def nasa_score(y_true, y_pred, scale=100):
    try:
        y_true_np = np.asarray(y_true).astype(np.float64) # Ensure float64 for precision
        y_pred_np = np.asarray(y_pred).astype(np.float64)
    except Exception as e:
        logging.error(f"Error converting inputs to numpy arrays for NASA score: {e}")
        return float('nan')
    if y_true_np.size == 0 or y_pred_np.size == 0:
        logging.warning("NASA score received empty array(s). Returning NaN.")
        return float('nan')
    if y_true_np.shape != y_pred_np.shape:
        logging.warning(f"NASA score shape mismatch: true {y_true_np.shape}, pred {y_pred_np.shape}. Returning NaN.")
        return float('nan')
    d = y_pred_np - y_true_np 
    score = np.zeros_like(d, dtype=np.float64)
    neg_mask = d < 0
    pos_mask = ~neg_mask
    neg_d_clipped = np.clip(d[neg_mask] * scale, -np.inf, -1e-9) 
    pos_d_clipped = np.clip(d[pos_mask] * scale, 1e-9, 700)      
    try:
        score[neg_mask] = np.exp(-neg_d_clipped / 13.0) - 1.0
        score[pos_mask] = np.exp(pos_d_clipped / 10.0) - 1.0
    except FloatingPointError as e:
         logging.error(f"FloatingPointError during NASA score exp calculation: {e}", exc_info=True)
         score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    max_penalty = 1e6
    if np.any(np.isnan(score)) or np.any(np.isinf(score)):
        logging.warning("NaN or Inf detected during NASA score calculation. Replacing faulty entries with max penalty.")
        score = np.nan_to_num(score, nan=max_penalty, posinf=max_penalty, neginf=-max_penalty)
    final_score = np.mean(score)
    if np.isnan(final_score) or np.isinf(final_score):
        logging.error(f"Final NASA score is NaN or Inf ({final_score}). Returning max penalty.")
        return max_penalty
    return final_score


# --- Combined Initial Pass ---
@torch.no_grad()
def initial_pass_and_precompute(
    model: CEM_Regression,
    dataloader: DataLoader,
    eval_metrics: Dict[str, torchmetrics.Metric],
    device: torch.device
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    n_concepts = model.n_concepts
    emb_size = model.emb_size
    for metric in eval_metrics.values(): metric.reset()

    true_concepts_list, pred_concepts_logits_list, true_labels_list, original_contexts_list, all_preds_cpu = [], [], [], [], []
    logging.info("Starting initial pass & precomputation...")
    pass_start_time = time.time()
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        try:
            x, y_rul, c_concepts = batch
            if x is None or y_rul is None or c_concepts is None or x.shape[0] == 0:
                 logging.warning(f"Skipping empty or invalid batch {i+1}/{num_batches}.")
                 continue
            x, y_rul_dev = x.to(device), y_rul.to(device)
            c_concepts = c_concepts.float() # Ensure concepts are float
        except Exception as e:
            logging.error(f"Error processing batch {i+1}/{num_batches}: {e}", exc_info=True)
            continue

        try:
            pre_c = model.pre_concept_model(x)
            batch_contexts = torch.empty((x.shape[0], n_concepts, 2 * emb_size), device=device, dtype=x.dtype)
            batch_probs = torch.empty((x.shape[0], n_concepts), device=device, dtype=x.dtype)
            for c_idx in range(n_concepts):
                if c_idx >= len(model.concept_context_generators) or (not model.shared_prob_gen and c_idx >= len(model.concept_prob_generators)):
                     logging.error(f"Missing generator for concept index {c_idx}. Filling with defaults.")
                     batch_contexts[:, c_idx, :] = 0.0; batch_probs[:, c_idx] = 0.5; continue
                context = model.concept_context_generators[c_idx](pre_c)
                prob_gen = model.concept_prob_generators[0] if model.shared_prob_gen else model.concept_prob_generators[c_idx]
                prob = torch.sigmoid(prob_gen(context))
                batch_contexts[:, c_idx, :] = context
                batch_probs[:, c_idx] = prob.squeeze(-1)

            embeddings = (batch_contexts[:, :, :emb_size] * batch_probs.unsqueeze(-1) +
                          batch_contexts[:, :, emb_size:] * (1 - batch_probs.unsqueeze(-1)))
            embeddings_flat = embeddings.view(embeddings.shape[0], -1)
            y_pred = model.c2y_model(embeddings_flat)
        except Exception as e:
            logging.error(f"Error during forward pass in batch {i+1}/{num_batches}: {e}", exc_info=True)
            continue

        y_pred_flat, y_rul_flat_dev = y_pred.squeeze(), y_rul_dev.squeeze()
        if y_pred_flat.ndim == 0: y_pred_flat = y_pred_flat.unsqueeze(0)
        if y_rul_flat_dev.ndim == 0: y_rul_flat_dev = y_rul_flat_dev.unsqueeze(0)
        if y_pred_flat.shape[0] != y_rul_flat_dev.shape[0]:
            logging.error(f"Shape mismatch preds/labels batch {i+1}/{num_batches}: {y_pred_flat.shape} vs {y_rul_flat_dev.shape}. Skipping metric update.")
            continue

        for name, metric in eval_metrics.items():
            try: metric.update(y_pred_flat.float(), y_rul_flat_dev.float())
            except Exception as e: logging.error(f"Error updating metric {name} batch {i+1}: {e}", exc_info=True)

        epsilon = 1e-7
        batch_probs_clamped = torch.clamp(batch_probs, epsilon, 1 - epsilon)
        batch_logits = torch.log(batch_probs_clamped / (1 - batch_probs_clamped))
        true_concepts_list.append(c_concepts.cpu())
        pred_concepts_logits_list.append(batch_logits.cpu())
        true_labels_list.append(y_rul.cpu())
        original_contexts_list.append(batch_contexts.cpu())
        all_preds_cpu.append(y_pred_flat.cpu())
        if (i + 1) % 100 == 0: logging.info(f" Initial pass: Processed batch {i+1}/{num_batches}")

    pass_end_time = time.time()
    logging.info(f"Initial pass loop finished in {pass_end_time - pass_start_time:.2f} seconds.")

    initial_metrics = {}
    logging.info("Computing final Step 0 metrics...")
    for name, metric in eval_metrics.items():
        try: initial_metrics[name] = metric.compute().item() if metric.update_count > 0 else float('nan')
        except Exception as e: logging.error(f"Could not compute metric {name}: {e}. Setting NaN."); initial_metrics[name] = float('nan')

    if 'mean_squared_error' in initial_metrics and not np.isnan(initial_metrics['mean_squared_error']):
        initial_metrics['rmse'] = np.sqrt(max(0, initial_metrics['mean_squared_error']))
    else: initial_metrics['rmse'] = float('nan')

    if not all_preds_cpu: initial_metrics['nasa_score'] = float('nan')
    else:
        try:
            all_preds_np = torch.cat(all_preds_cpu).numpy() if all_preds_cpu else np.array([])
            all_targets_np = torch.cat(true_labels_list).squeeze().numpy() if true_labels_list else np.array([])
            if all_preds_np.size > 0 and all_targets_np.size > 0: initial_metrics['nasa_score'] = nasa_score(all_targets_np, all_preds_np)
            else: initial_metrics['nasa_score'] = float('nan')
        except Exception as e: logging.error(f"Error calculating NASA score Step 0: {e}", exc_info=True); initial_metrics['nasa_score'] = float('nan')
    logging.info(f"Step 0 Metrics: {initial_metrics}")

    logging.info("Concatenating precomputed tensors...")
    if not true_concepts_list:
        logging.error("No valid batches processed. Cannot proceed.")
        return initial_metrics, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    try:
        true_concepts = torch.cat(true_concepts_list, dim=0)
        pred_concepts_logits = torch.cat(pred_concepts_logits_list, dim=0)
        true_labels = torch.cat(true_labels_list, dim=0)
        original_contexts = torch.cat(original_contexts_list, dim=0)
    except Exception as e:
        logging.error(f"Error concatenating tensors: {e}", exc_info=True)
        return initial_metrics, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    logging.info(f"Precomputation finished. Shapes: Concepts: {true_concepts.shape}, Logits: {pred_concepts_logits.shape}, Labels: {true_labels.shape}, Contexts: {original_contexts.shape}")
    del true_concepts_list, pred_concepts_logits_list, true_labels_list, original_contexts_list, all_preds_cpu
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return initial_metrics, true_concepts, pred_concepts_logits, true_labels, original_contexts


# --- Optimized Intervention Application ---
@torch.no_grad()
def get_intervened_embeddings_optimized(
        true_concepts: torch.Tensor, # [N, C], CPU
        pred_concepts_logits: torch.Tensor, # [N, C], CPU
        intervention_mask: torch.Tensor, # [N, C], bool, CPU
        original_contexts: torch.Tensor, # [N, C, 2*Emb], CPU
        intervention_format: str,
        probs_min: float,
        probs_max: float,
        emb_size: int,
        device: torch.device,
        include_uncertain: bool = False
    ) -> torch.Tensor:
    intervention_mask_dev = intervention_mask.to(device)
    true_concepts_float_dev = true_concepts.float().to(device)
    pred_concepts_logits_dev = pred_concepts_logits.to(device)
    original_contexts_dev = original_contexts.to(device)
    pred_probs_dev = torch.sigmoid(pred_concepts_logits_dev)

    if intervention_format == "BINARY":
        intervened_prob_val = true_concepts_float_dev
    elif intervention_format == "PROBS":
        dtype = pred_probs_dev.dtype
        intervened_prob_val = torch.where(true_concepts_float_dev == 1.0,
                                          torch.tensor(probs_max, device=device, dtype=dtype),
                                          torch.tensor(probs_min, device=device, dtype=dtype))
    else: raise ValueError(f'Intervention format {intervention_format} not supported.')

    intervened_probs = torch.where(intervention_mask_dev, intervened_prob_val, pred_probs_dev)
    context_pos = original_contexts_dev[:, :, :emb_size]
    context_neg = original_contexts_dev[:, :, emb_size:]
    mixed_embeddings = (context_pos * intervened_probs.unsqueeze(-1) +
                        context_neg * (1.0 - intervened_probs.unsqueeze(-1)))
    return mixed_embeddings.view(mixed_embeddings.shape[0], -1)



# --- Optimized Evaluation Step Function ---
@torch.no_grad()
def evaluate_intervention_step_regression_optimized(
        ctoy_model: nn.Module,
        intervened_embeddings: torch.Tensor, # [N, C*Emb], DEVICE
        true_labels: torch.Tensor,         # [N, 1] or [N], CPU
        metrics: Dict[str, torchmetrics.Metric], # Metrics on DEVICE
        batch_size: int,
        device: torch.device) -> Dict[str, float]:
    ctoy_model.eval()
    n_samples = intervened_embeddings.shape[0]
    if n_samples == 0:
        logging.warning("evaluate_intervention_step received empty embeddings. Returning NaN metrics.")
        return {name: float('nan') for name in metrics} | {'nasa_score': float('nan'), 'rmse': float('nan')}
    for metric in metrics.values(): metric.reset()

    true_labels_dev = true_labels.to(device).squeeze()
    if true_labels_dev.ndim == 0: true_labels_dev = true_labels_dev.unsqueeze(0)

    all_preds_cpu = []
    for i in range(0, n_samples, batch_size):
        b_start, b_end = i, min(i + batch_size, n_samples)
        if b_start >= b_end: continue
        embeddings_batch_dev = intervened_embeddings[b_start:b_end]
        labels_batch_dev = true_labels_dev[b_start:b_end]
        try:
            predictions_batch_dev = ctoy_model(embeddings_batch_dev).squeeze()
        except Exception as e:
            logging.error(f"Error in CtoY forward pass eval step (batch {i//batch_size}): {e}", exc_info=True)
            continue
        if predictions_batch_dev.ndim == 0: predictions_batch_dev = predictions_batch_dev.unsqueeze(0)
        if labels_batch_dev.ndim == 0: labels_batch_dev = labels_batch_dev.unsqueeze(0) # Should be handled
        if predictions_batch_dev.shape[0] != labels_batch_dev.shape[0]:
            logging.error(f"Shape mismatch eval step batch {i//batch_size}: {predictions_batch_dev.shape} vs {labels_batch_dev.shape}")
            continue
        for name, metric in metrics.items():
            try: metric.update(predictions_batch_dev.float(), labels_batch_dev.float())
            except Exception as e: logging.error(f"Error updating metric {name} eval step (batch {i//batch_size}): {e}", exc_info=True)
        all_preds_cpu.append(predictions_batch_dev.cpu())

    eval_results = {}
    for name, metric in metrics.items():
        try: eval_results[name] = metric.compute().item() if metric.update_count > 0 else float('nan')
        except Exception as e: logging.error(f"Could not compute metric {name} eval step: {e}. Setting NaN."); eval_results[name] = float('nan')

    if not all_preds_cpu: eval_results['nasa_score'] = float('nan')
    else:
        try:
            all_preds_np = torch.cat(all_preds_cpu).numpy() if all_preds_cpu else np.array([])
            all_targets_np = true_labels.squeeze().numpy()
            if all_preds_np.size > 0 and all_targets_np.size > 0: eval_results['nasa_score'] = nasa_score(all_targets_np, all_preds_np)
            else: eval_results['nasa_score'] = float('nan')
        except Exception as e: logging.error(f"Error calculating NASA score eval step: {e}", exc_info=True); eval_results['nasa_score'] = float('nan')

    if 'mean_squared_error' in eval_results and not np.isnan(eval_results['mean_squared_error']):
        eval_results['rmse'] = np.sqrt(max(0, eval_results['mean_squared_error']))
    else: eval_results['rmse'] = float('nan')
    if 'mean_absolute_error' not in eval_results: eval_results['mean_absolute_error'] = float('nan')

    return eval_results


# --- Main Intervention Evaluation Function (Optimized - Unchanged Logic) ---
def evaluate_intervention_optimized(
    cem_model: CEM_Regression,
    ds_val: DataLoader, # DataLoader for the (potentially subsampled) evaluation set
    eval_metrics: Dict[str, torchmetrics.Metric],
    policy_type: PolicyType,
    policy_args: Dict[str, Any],
    n_steps: int,
    device: torch.device,
    batch_size: int = 1024,
    intervention_format: str = "BINARY",
    include_uncertain: bool = False,
    ) -> Tuple[List[Dict[str, float]], List[Optional[List[str]]]]:
    
    required_keys = {'concept_names', 'budget', 'info_single_concepts', 'info_structured_costs', 'emb_size'}
    if policy_type in [PolicyType.GREEDY_CONCEPTS, PolicyType.GREEDY_GROUPS, PolicyType.OPTIMIZED]:
        required_keys.update({'concept_metric', 'label_metric', 'label_metric_weight'})
    if missing_keys := required_keys - policy_args.keys(): raise ValueError(f"Missing keys: {missing_keys}")

    concept_names, n_concepts = policy_args['concept_names'], len(policy_args['concept_names'])
    emb_size, budget = policy_args['emb_size'], policy_args['budget']
    info_single_concepts, info_structured_costs = policy_args['info_single_concepts'], policy_args['info_structured_costs']
    concept_metric_enum, label_metric_enum = policy_args.get('concept_metric'), policy_args.get('label_metric')
    label_metric_weight = policy_args.get('label_metric_weight')

    cem_model.eval()
    logging.info("--- Performing Initial Pass (Step 0 Eval + Precomputation) ---")
    initial_metrics, true_concepts, pred_concepts_logits, true_labels, original_contexts = \
        initial_pass_and_precompute(cem_model, ds_val, eval_metrics, device)

    if true_concepts.numel() == 0: # Check if precomputation returned valid data
        logging.error("Precomputation failed or dataset empty. Aborting intervention.")
        nan_metrics = {name: float('nan') for name in eval_metrics} | {'rmse': float('nan'), 'nasa_score': float('nan')}
        return [nan_metrics] * (n_steps + 1), [None] + [[]] * n_steps

    intervention_metrics_history = [initial_metrics]
    concepts_revealed_at_step = [None]
    logging.info(f"Step 0 evaluation metrics: {initial_metrics}")
    n_samples = true_concepts.shape[0]

    probs_min, probs_max = 0.05, 0.95
    if intervention_format == "PROBS":
        pred_probs_cpu = torch.sigmoid(pred_concepts_logits).flatten()
        if pred_probs_cpu.numel() > 10:
            try:
                q_min, q_max = torch.quantile(pred_probs_cpu, torch.tensor([0.05, 0.95], device='cpu')).tolist()
                probs_min, probs_max = max(0.0, min(q_min, 1.0)), max(0.0, min(q_max, 1.0))
                if probs_min >= probs_max - 1e-6 : # Add tolerance
                     logging.warning(f"Prob quantiles invalid ({probs_min:.4f}>={probs_max:.4f}). Using defaults.")
                     probs_min, probs_max = 0.05, 0.95
                else: logging.info(f"Intervention range (Probs): [{probs_min:.4f}, {probs_max:.4f}]")
            except Exception as e: logging.warning(f"Could not calc prob quantiles: {e}. Using defaults."); probs_min, probs_max = 0.05, 0.95
        else: logging.warning("Not enough data for prob quantiles. Using defaults."); probs_min, probs_max = 0.05, 0.95
        del pred_probs_cpu
    elif intervention_format == "BINARY": logging.info("Using BINARY intervention format.")

    intervention_mask = torch.zeros((n_samples, n_concepts), dtype=torch.bool, device='cpu')
    concept_metric_fnc = get_metric_fn(concept_metric_enum) if concept_metric_enum else None
    label_metric_fnc = get_metric_fn(label_metric_enum) if label_metric_enum else None

    c_uncertainty_scores = None
    if policy_type in [PolicyType.GREEDY_CONCEPTS, PolicyType.GREEDY_GROUPS, PolicyType.OPTIMIZED] and concept_metric_fnc:
        logging.info("Calculating concept uncertainty scores...")
        c_uncertainty_scores = concept_metric_fnc(pred_concepts_logits.cpu())
        c_uncertainty_scores = torch.nan_to_num(c_uncertainty_scores, nan=0.0).cpu()
        logging.info("Calculated concept uncertainty scores.")

    # --- Intervention Loop ---
    for s in range(1, n_steps + 1):
        logging.info(f"--- Intervention Step {s}/{n_steps} ---")
        step_start_time = time.time()
        concept_importance_scores = torch.zeros_like(pred_concepts_logits, device='cpu')

        # --- 1. Calculate Label Importance (if needed) ---
        if policy_type in [PolicyType.GREEDY_CONCEPTS, PolicyType.GREEDY_GROUPS, PolicyType.OPTIMIZED] and \
           label_metric_fnc is not None and label_metric_weight is not None and label_metric_weight > 1e-6:
            logging.info(f" Step {s}: Calculating label importance scores...")
            calc_start_time = time.time()
            value_0, value_1 = (0.0, 1.0) if intervention_format == "BINARY" else (probs_min, probs_max)
            try:
                current_embeddings_dev = get_intervened_embeddings_optimized(true_concepts, pred_concepts_logits, intervention_mask, original_contexts, intervention_format, probs_min, probs_max, emb_size, device, include_uncertain)
                pred_labels_0_dev = torch.zeros((n_samples, n_concepts), device=device, dtype=current_embeddings_dev.dtype)
                pred_labels_1_dev = torch.zeros((n_samples, n_concepts), device=device, dtype=current_embeddings_dev.dtype)
                current_rul_pred_dev = torch.zeros(n_samples, device=device, dtype=current_embeddings_dev.dtype)

                original_contexts_dev = original_contexts.to(device)
                pred_concepts_logits_dev = pred_concepts_logits.to(device)
                true_concepts_float_dev = true_concepts.float().to(device)
                context_pos_dev, context_neg_dev = original_contexts_dev[:, :, :emb_size], original_contexts_dev[:, :, emb_size:]
                intervention_mask_dev = intervention_mask.to(device)

                for i in range(0, n_samples, batch_size): # Calc current preds
                    b_start, b_end = i, min(i + batch_size, n_samples)
                    if b_start >= b_end: continue
                    try: current_rul_pred_dev[b_start:b_end] = cem_model.c2y_model(current_embeddings_dev[b_start:b_end]).squeeze()
                    except Exception as e: logging.error(f"Err pred current RUL batch {i//batch_size}: {e}", exc_info=True); current_rul_pred_dev[b_start:b_end] = 0.0

                dtype_probs = pred_concepts_logits_dev.dtype # Match dtype for tensors
                if intervention_format == "BINARY": intervened_val_for_mask = true_concepts_float_dev
                else: intervened_val_for_mask = torch.where(true_concepts_float_dev == 1.0, torch.tensor(probs_max, device=device, dtype=dtype_probs), torch.tensor(probs_min, device=device, dtype=dtype_probs))

                for j in range(n_concepts): # Calc hypothetical preds per concept
                    current_probs_dev = torch.sigmoid(pred_concepts_logits_dev)
                    mask_except_j = intervention_mask_dev.clone(); mask_except_j[:, j] = False
                    current_intervened_probs_except_j = torch.where(mask_except_j, intervened_val_for_mask, current_probs_dev)
                    probs_0 = current_intervened_probs_except_j.clone(); probs_1 = current_intervened_probs_except_j.clone()
                    probs_0[:, j] = value_0; probs_1[:, j] = value_1

                    embed_0_j = (context_pos_dev * probs_0.unsqueeze(-1) + context_neg_dev * (1.0 - probs_0.unsqueeze(-1)))
                    embed_1_j = (context_pos_dev * probs_1.unsqueeze(-1) + context_neg_dev * (1.0 - probs_1.unsqueeze(-1)))
                    embed_0_j_flat, embed_1_j_flat = embed_0_j.view(n_samples, -1), embed_1_j.view(n_samples, -1)

                    for i in range(0, n_samples, batch_size): # Batched prediction
                        b_start, b_end = i, min(i + batch_size, n_samples)
                        if b_start >= b_end: continue
                        try:
                            pred_0 = cem_model.c2y_model(embed_0_j_flat[b_start:b_end]).squeeze()
                            pred_1 = cem_model.c2y_model(embed_1_j_flat[b_start:b_end]).squeeze()
                            pred_labels_0_dev[b_start:b_end, j] = pred_0 if pred_0.ndim > 0 else pred_0.unsqueeze(0)
                            pred_labels_1_dev[b_start:b_end, j] = pred_1 if pred_1.ndim > 0 else pred_1.unsqueeze(0)
                        except Exception as e: logging.error(f"Err pred hypo RUL c={j} b={i//batch_size}: {e}", exc_info=True); pred_labels_0_dev[b_start:b_end, j]=0.0; pred_labels_1_dev[b_start:b_end, j]=0.0
                    del embed_0_j, embed_1_j, embed_0_j_flat, embed_1_j_flat, probs_0, probs_1, mask_except_j # Clean inside loop

                current_rul_pred_dev_exp = current_rul_pred_dev.unsqueeze(-1) if current_rul_pred_dev.ndim == 1 else current_rul_pred_dev
                batch_importance_dev = label_metric_fnc(pred_concepts_logits_dev, current_rul_pred_dev_exp, pred_labels_0_dev, pred_labels_1_dev)
                concept_importance_scores = batch_importance_dev.cpu()
                calc_end_time = time.time()
                logging.info(f" Label importance calculation took {calc_end_time - calc_start_time:.2f} seconds.")

            except Exception as e:
                 logging.error(f"Error calculating label importance in step {s}: {e}. Skipping.", exc_info=True)
                 concept_importance_scores.fill_(0.0) # Ensure zero if failed
            finally: # Ensure cleanup happens
                 if 'current_embeddings_dev' in locals(): del current_embeddings_dev
                 if 'pred_labels_0_dev' in locals(): del pred_labels_0_dev, pred_labels_1_dev, current_rul_pred_dev
                 if 'original_contexts_dev' in locals(): del original_contexts_dev, pred_concepts_logits_dev, true_concepts_float_dev, intervention_mask_dev
                 if 'context_pos_dev' in locals(): del context_pos_dev, context_neg_dev
                 if 'intervened_val_for_mask' in locals(): del intervened_val_for_mask
                 if 'batch_importance_dev' in locals(): del batch_importance_dev
                 if torch.cuda.is_available(): torch.cuda.empty_cache()


        # --- 2. Combine Scores and Select Concepts ---
        selection_start_time = time.time()
        final_mask_from_policy = None; step_concepts_revealed = []
        try:
            combined_scores = torch.zeros_like(pred_concepts_logits, device='cpu') # Default to zeros
            # Determine scores based on policy type and available metrics
            if policy_type in [PolicyType.GREEDY_CONCEPTS, PolicyType.GREEDY_GROUPS, PolicyType.OPTIMIZED]:
                use_uncertainty = c_uncertainty_scores is not None
                use_importance = torch.any(concept_importance_scores != 0)
                weight_valid = label_metric_weight is not None and 1e-6 < label_metric_weight < 1.0 - 1e-6

                if use_uncertainty and (not use_importance or (label_metric_weight is not None and label_metric_weight <= 1e-6)):
                    logging.info(f"Step {s}: Using only concept uncertainty.")
                    combined_scores = c_uncertainty_scores.cpu().clone()
                elif use_importance and (not use_uncertainty or (label_metric_weight is not None and label_metric_weight >= 1.0 - 1e-6)):
                     logging.info(f"Step {s}: Using only label importance.")
                     combined_scores = concept_importance_scores.cpu().clone()
                elif use_uncertainty and use_importance and weight_valid:
                    logging.info(f"Step {s}: Combining uncertainty (w={1-label_metric_weight:.2f}) & importance (w={label_metric_weight:.2f}).")
                    combined_scores = (1.0 - label_metric_weight) * c_uncertainty_scores.cpu() + label_metric_weight * concept_importance_scores.cpu()
                elif use_uncertainty: # Fallback if importance failed or weight is extreme
                    logging.warning(f"Step {s}: Falling back to uncertainty scores only.")
                    combined_scores = c_uncertainty_scores.cpu().clone()
                else: # Worst case: no uncertainty, no importance
                    logging.warning(f"Step {s}: No valid scores available for greedy policy. Using zeros.")
                    # Policy will likely fail or do nothing meaningful

                combined_scores = torch.nan_to_num(combined_scores, nan=0.0, posinf=1e9, neginf=-1e9)
                policy_fn_name = policy_type.value 
                policy_fn = getattr(baseline_policies, policy_fn_name)
                pparams = dict( global_intervention_mask=intervention_mask.clone(), concept_values=combined_scores, concept_names=concept_names, budget=budget, include_uncertain=include_uncertain, concept_uncertainty=c_uncertainty_scores.cpu() if c_uncertainty_scores is not None else torch.zeros_like(pred_concepts_logits, device='cpu'))
                if 'concepts' in policy_type.value: pparams['concept_info'] = info_single_concepts
                if 'groups' in policy_type.value or policy_type == PolicyType.OPTIMIZED: pparams['structured_costs'] = info_structured_costs
                logging.info(f" Applying policy '{policy_fn_name}'...")
                final_mask_from_policy, step_concepts_revealed = policy_fn(**pparams)

            else: # Random policies
                policy_fn_name = policy_type.value
                policy_fn = getattr(baseline_policies, policy_fn_name)
                pparams = dict( global_intervention_mask=intervention_mask.clone(), concept_names=concept_names, budget=budget, include_uncertain=include_uncertain, concept_uncertainty=torch.zeros_like(pred_concepts_logits, device='cpu'))
                if 'concepts' in policy_type.value: pparams['concept_info'] = info_single_concepts
                if 'groups' in policy_type.value: pparams['structured_costs'] = info_structured_costs
                logging.info(f" Applying policy '{policy_fn_name}'...")
                final_mask_from_policy, step_concepts_revealed = policy_fn(**pparams)

        except Exception as e:
            logging.error(f"Error during policy application step {s}: {e}. Skipping intervention.", exc_info=True)
            final_mask_from_policy = intervention_mask.clone(); step_concepts_revealed = [] # No change

        selection_end_time = time.time()
        logging.info(f" Policy selection took {selection_end_time - selection_start_time:.2f} seconds.")

        # --- 3. Update Global Intervention State ---
        if final_mask_from_policy is None:
            logging.error(f"Step {s}: Policy returned None mask. Stopping.")
            concepts_revealed_at_step.append([])
            intervention_metrics_history.append(intervention_metrics_history[-1])
            break

        final_mask_from_policy = final_mask_from_policy.cpu()
        if torch.equal(intervention_mask, final_mask_from_policy):
            logging.warning(f"Step {s}: Mask unchanged. Budget likely exhausted or no valid interventions. Stopping.")
            concepts_revealed_at_step.append([])
            intervention_metrics_history.append(intervention_metrics_history[-1])
            break
        else:
            intervention_mask = final_mask_from_policy
            revealed_count = len(step_concepts_revealed) if step_concepts_revealed else 0
            logging.info(f"Step {s}: Mask updated. Revealed {revealed_count} concepts. Total intervened: {intervention_mask.sum().item()}/{n_samples*n_concepts}.")
            concepts_revealed_at_step.append(step_concepts_revealed)


        # --- 4. Evaluate State After Intervention ---
        eval_start_time = time.time()
        try:
            intervened_embeddings_dev = get_intervened_embeddings_optimized(true_concepts, pred_concepts_logits, intervention_mask, original_contexts, intervention_format, probs_min, probs_max, emb_size, device, include_uncertain)
            step_metrics = evaluate_intervention_step_regression_optimized(cem_model.c2y_model, intervened_embeddings_dev, true_labels, eval_metrics, batch_size, device)
            intervention_metrics_history.append(step_metrics)
            eval_end_time = time.time()
            logging.info(f"Step {s} evaluation metrics: {step_metrics}")
            logging.info(f" Step {s} evaluation took {eval_end_time - eval_start_time:.2f} seconds.")
            del intervened_embeddings_dev
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error during evaluation step {s}: {e}. Appending previous metrics.", exc_info=True)
            intervention_metrics_history.append(intervention_metrics_history[-1])
            if 'intervened_embeddings_dev' in locals(): del intervened_embeddings_dev
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        step_end_time = time.time()
        logging.info(f"--- Intervention Step {s} completed in {step_end_time - step_start_time:.2f} seconds ---")
        if torch.all(intervention_mask):
            logging.info(f"Step {s}: All concepts intervened. Stopping.")
            break

    # --- Pad results if loop stopped early ---
    num_steps_run = len(intervention_metrics_history) - 1
    if num_steps_run < n_steps:
         logging.warning(f"Loop stopped early after {num_steps_run}/{n_steps} steps. Padding results.")
         last_metrics = intervention_metrics_history[-1]
         last_revealed = concepts_revealed_at_step[-1] if concepts_revealed_at_step else []
         pad_count = (n_steps + 1) - len(intervention_metrics_history)
         intervention_metrics_history.extend([last_metrics] * pad_count)
         concepts_revealed_at_step.extend([last_revealed] * pad_count)
    intervention_metrics_history = intervention_metrics_history[:n_steps + 1]
    concepts_revealed_at_step = concepts_revealed_at_step[:n_steps + 1]

    return intervention_metrics_history, concepts_revealed_at_step


# --- Main Execution Block ---
def main(argv: Sequence[str], run_seed: Optional[int] = None):
    # if len(argv) > 1: raise app.UsageError('Too many command-line arguments.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print(FLAGS.seed)
    print(run_seed)

    seed = FLAGS.seed + run_seed if run_seed is not None else FLAGS.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        logging.info(f"CUDA available. CUDA_VISIBLE_DEVICES='{cuda_devices if cuda_devices else 'Not Set'}'")
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try: seed_everything(seed, workers=True); logging.info(f"Applied seed {seed} via seed_everything.")
    except Exception as e: logging.warning(f"seed_everything failed: {e}.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    config = FLAGS
    start_time = time.time()

    # --- Log Core Configuration ---
    logging.info("--- Core Configuration ---")
    logging.info(f" Seed: {config.seed}")
    logging.info(f" Dataset: {config.dataset}, N_DS: {config.n_ds}, Units: {config.test_units if config.test_units else 'All'}")
    logging.info(f" Architecture: {config.architecture}, C Extractor: {config.c_extractor_arch}, Emb Size: {config.emb_size}")
    logging.info(f" Concepts: {len(config.concepts)} total, Binary: {config.binary_concepts}")
    logging.info(f" Window: {config.window_size}, Batch: {config.batch_size}")
    # Log the correct subsampling flag
    logging.info(f" Eval Subsample Size (windows): {config.eval_subsample_size if config.eval_subsample_size else 'Full Dataset'}")
    logging.info(f" Policy: {config.policy_type}, Budget: {config.budget}, Steps: {config.n_intervention_steps}")
    if 'greedy' in config.policy_type or 'optimized' in config.policy_type:
        logging.info(f"  Metrics: Concept='{config.concept_metric}', Label='{config.label_metric}', Label Weight={config.label_metric_weight}")
    logging.info(f" Intervention Format: {config.intervention_format}")
    logging.info(f" Checkpoint: {config.checkpoint if config.checkpoint else 'Derived'}")
    logging.info(f" Output Dir: {config.output_dir}")
    logging.info("--------------------------")

    # --- Load Data ---
    if config.dataset == "N-CMAPSS": DatasetClass = NCMAPSSDataset
    elif config.dataset == "CMAPSS": DatasetClass = CMAPSSDataset
    else: raise ValueError(f"Unsupported dataset: {config.dataset}")

    eval_datasets = []
    logging.info("--- Loading Evaluation Data ---")
    for n_ds in config.n_ds:
        current_test_units = [int(u) for u in config.test_units] if config.test_units else None
        logging.info(f" Processing DS '{n_ds}', Target Units: {current_test_units if current_test_units else 'All'}")
        dataset_params = dict( path=config.data_path, n_DS=n_ds, units=current_test_units, concepts=config.concepts, binary_concepts=config.binary_concepts, RUL=config.rul_type, include_healthy=True, subsampling_rate=config.downsample, window_size=config.window_size, stride=config.stride, scaling=config.scaling, combined_concepts=config.combined_concepts, )
        loaded_ds = False
        for mode in ["test", "val", "train"]:
             try:
                 logging.info(f"  Attempting mode='{mode}'...")
                 ds_instance = DatasetClass(**dataset_params, mode=mode)
                 if len(ds_instance) > 0:
                     eval_datasets.append(ds_instance); logging.info(f"  Loaded mode='{mode}' ({len(ds_instance)} samples)."); loaded_ds = True; break
                 else: logging.warning(f"  Mode='{mode}' empty.")
             except FileNotFoundError: logging.warning(f"  File not found for mode='{mode}'.")
             except Exception as e: logging.error(f"  Error loading mode '{mode}': {e}", exc_info=True)
        if not loaded_ds: logging.warning(f"Could not load any data for DS '{n_ds}'.")

    if not eval_datasets: raise ValueError("No evaluation data loaded.")
    eval_ds_combined = ConcatDataset(eval_datasets)
    total_samples_available = len(eval_ds_combined)
    logging.info(f"Combined datasets. Total samples available: {total_samples_available}")

    # --- Subsampling Logic (Random Windows) ---
    eval_ds_for_loader = eval_ds_combined # Start with the full dataset
    if config.eval_subsample_size is not None and config.eval_subsample_size > 0:
        if total_samples_available == 0:
             logging.warning("Subsampling requested, but no samples available.")
        else:
            subset_size = min(config.eval_subsample_size, total_samples_available)
            logging.info(f"Subsampling {subset_size} random windows (requested {config.eval_subsample_size}, available {total_samples_available}).")
            # Use numpy random state seeded earlier for reproducibility
            subset_indices = np.random.choice(np.arange(total_samples_available), size=subset_size, replace=False)
            eval_ds_subset = Subset(eval_ds_combined, subset_indices.tolist())
            eval_ds_for_loader = eval_ds_subset # Use the subset
            logging.info(f"Created evaluation subset with {len(eval_ds_for_loader)} samples.")
    else:
        logging.info("Using the full combined evaluation dataset.")
    

    if len(eval_ds_for_loader) == 0: raise ValueError("Final evaluation dataset is empty.")

    eval_dl = DataLoader( eval_ds_for_loader, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False )
    logging.info(f"Created DataLoader with {len(eval_ds_for_loader)} samples.")

    # --- Get Concept Names (Robust handling) ---
    concept_names = config.concepts # Default to flags
    try:
        dataset_to_inspect = eval_ds_for_loader
        while isinstance(dataset_to_inspect, (Subset, ConcatDataset)):
             if isinstance(dataset_to_inspect, Subset): dataset_to_inspect = dataset_to_inspect.dataset
             elif isinstance(dataset_to_inspect, ConcatDataset):
                  if not dataset_to_inspect.datasets: raise AttributeError("ConcatDataset is empty")
                  dataset_to_inspect = dataset_to_inspect.datasets[0] # Inspect the first one
        # Now dataset_to_inspect should be the base NCMAPSSDataset/CMAPSSDataset
        if hasattr(dataset_to_inspect, 'concepts') and hasattr(dataset_to_inspect.concepts, 'columns'):
            concept_names = dataset_to_inspect.concepts.columns.tolist()
        else: raise AttributeError("Base dataset lacks 'concepts.columns'")
    except Exception as e:
        logging.warning(f"Could not get concepts from data ({e}). Using flags: {concept_names}")
    n_concepts = len(concept_names)
    if set(concept_names) != set(config.concepts):
         logging.warning(f"Concept mismatch data vs flags. Using from data: {concept_names}")
    else: logging.info(f"Using concepts: {concept_names} ({n_concepts} total)")
    info_structured_costs, info_single_concepts = get_cmapss_concept_costs()
    if set(concept_names) != set(info_single_concepts.keys()):
         logging.warning("Concept mismatch data vs costs. Filtering costs.")
         info_single_concepts = {k: v for k, v in info_single_concepts.items() if k in concept_names}
         
    # --- Load CEM Model ---
    model_saved_path = config.checkpoint
    if model_saved_path is None or not os.path.exists(model_saved_path):
        potential_ckpt_dir = os.path.join(config.output_dir, f"split_{config.split}", "checkpoints")
        if not os.path.isdir(potential_ckpt_dir): potential_ckpt_dir = os.path.join(config.output_dir, "checkpoints")
        if not os.path.isdir(potential_ckpt_dir): potential_ckpt_dir = config.output_dir
        path_pt = os.path.join(potential_ckpt_dir, f"model_seed_{config.split}{config.checkpoint_suffix}")
        path_ckpt = os.path.join(potential_ckpt_dir, "last.ckpt")
        logging.warning(f"Checkpoint '{model_saved_path if model_saved_path else 'None'}' not found/provided.")
        logging.info(f"Searching in {potential_ckpt_dir} for '{os.path.basename(path_pt)}' or '{os.path.basename(path_ckpt)}'")
        if os.path.exists(path_pt): model_saved_path = path_pt
        elif os.path.exists(path_ckpt): model_saved_path = path_ckpt
        else:
            found_fallback = False
            if os.path.isdir(potential_ckpt_dir):
                 files = [f for f in os.listdir(potential_ckpt_dir) if f.endswith(('.pt', '.ckpt'))]
                 if files:
                      split_files = [f for f in files if f"split_{config.split}" in f or f"seed_{config.split}" in f]
                      files_to_use = split_files if split_files else files
                      files_to_use.sort()
                      model_saved_path = os.path.join(potential_ckpt_dir, files_to_use[-1])
                      logging.warning(f"Using fallback checkpoint: {model_saved_path}")
                      found_fallback = True
            if not found_fallback:
                 checked = [config.checkpoint if config.checkpoint else "None provided", path_pt, path_ckpt, f"{potential_ckpt_dir}/*.pt/ckpt"]
                 raise FileNotFoundError(f"Could not find model checkpoint. Checked: {', '.join(checked)}")

    logging.info(f"Attempting to load model from: {model_saved_path}")
    if 'cnn' in config.c_extractor_arch.lower(): backbone_arch = latent_cnn_code_generator_model
    elif 'mlp' in config.c_extractor_arch.lower(): backbone_arch = latent_mlp_code_generator_model
    else: raise ValueError(f"Unknown c_extractor_arch: {config.c_extractor_arch}")

    model = None
    try: # Try load_from_checkpoint
        if config.architecture == "ConceptEmbeddingModel":
             model_args = dict( n_concepts=n_concepts, n_tasks=1, emb_size=config.emb_size, c_extractor_arch=backbone_arch, training_intervention_prob=0.0, intervention_idxs=[], intervention_policy=None, active_intervention_values=None, inactive_intervention_values=None, embed_dim=config.emb_size*2, # Common default, check model code
                                extra_dims=0, concept_loss_weight=0.0, task_loss_weight=1.0, learning_rate=1e-3, weight_decay=0.0, optimizer="adam", momentum=0.9, shared_prob_gen=False, # Match training!
                                binary_concepts=config.binary_concepts, 
                               )
             model = CEM_Regression.load_from_checkpoint( model_saved_path, map_location=device, strict=False, **model_args )
             logging.info("Model loaded via load_from_checkpoint (strict=False).")
        else: raise NotImplementedError(f"Loading not implemented for {config.architecture}")
    except Exception as e1:
        logging.warning(f"load_from_checkpoint failed: {e1}. Attempting manual load...")
        try: 
            model = CEM_Regression(**model_args) 
            checkpoint_data = torch.load(model_saved_path, map_location=device)
            if 'state_dict' in checkpoint_data: state_dict = checkpoint_data['state_dict']
            elif 'model_state_dict' in checkpoint_data: state_dict = checkpoint_data['model_state_dict']
            elif isinstance(checkpoint_data, dict): state_dict = checkpoint_data
            else: raise ValueError("Unrecognized checkpoint format.")
            new_state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()} # Basic prefix cleaning
            load_result = model.load_state_dict(new_state_dict, strict=False)
            logging.info(f"Manual load (strict=False): Missing={load_result.missing_keys}, Unexpected={load_result.unexpected_keys}")
            significant_missing = [k for k in load_result.missing_keys if not ('running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k)]
            if load_result.unexpected_keys or significant_missing:
                 raise RuntimeError(f"Manual load failed: Unexpected={load_result.unexpected_keys}, Missing={significant_missing}")
            logging.info("Manual state_dict loading successful.")
        except Exception as e2:
            logging.error(f"Manual state_dict loading failed: {e2}", exc_info=True)
            raise RuntimeError(f"Could not load model from '{model_saved_path}'") from e2

    if model is None: raise RuntimeError("Model loading failed.")
    model = model.to(device); model.eval()
    logging.info("Model loaded and moved to device.")
    try: logging.info(f"Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    except: pass


    # --- Prepare Evaluation Metrics ---
    eval_metrics_torch = { 'mean_squared_error': torchmetrics.MeanSquaredError().to(device), 'mean_absolute_error': torchmetrics.MeanAbsoluteError().to(device), }
    logging.info("Initialized TorchMetrics.")


    # --- Run Intervention Evaluation ---
    policy_args = { 'concept_names': concept_names, 'budget': config.budget, 'emb_size': config.emb_size, 'concept_metric': config.concept_metric, 'label_metric': config.label_metric, 'label_metric_weight': config.label_metric_weight, 'info_single_concepts': info_single_concepts, 'info_structured_costs': info_structured_costs, }
    logging.info("--- Starting Intervention Evaluation Loop ---")
    intervention_metrics_history, concepts_revealed_history = evaluate_intervention_optimized(
        cem_model=model, ds_val=eval_dl, eval_metrics=eval_metrics_torch, policy_type=PolicyType(config.policy_type.lower()), policy_args=policy_args, n_steps=config.n_intervention_steps, device=device, batch_size=config.batch_size, intervention_format=config.intervention_format, include_uncertain=config.include_uncertain_in_intervention,
    )
    logging.info("--- Intervention Evaluation Finished ---")


    # --- Save Results (Robust handling) ---
    # Custom JSON encoder (defined below or imported)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else None # Return None for NaN/Inf
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
            elif isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
            elif isinstance(obj, torch.device): return str(obj)
            elif isinstance(obj, type): return obj.__name__
            elif isinstance(obj, enum.Enum): return obj.name # Handle enums
            # Avoid infinite recursion for complex objects, just use repr
            # elif hasattr(obj, '__dict__'): return obj.__dict__
            try: return super(NpEncoder, self).default(obj)
            except TypeError: return repr(obj) # Final fallback

    serializable_config = {}
    for name, value in config.flag_values_dict().items():
        try: json.dumps({name: value}, cls=NpEncoder); serializable_config[name] = value
        except TypeError: serializable_config[name] = repr(value) # Fallback if direct serialization fails even with encoder

    n_samples_evaluated = len(eval_ds_for_loader) # Get actual evaluated count

    results = { 'config': serializable_config, 'metrics_history': intervention_metrics_history, 'concepts_revealed_history': concepts_revealed_history, 'concept_names': concept_names, 'n_samples_evaluated': n_samples_evaluated, 'evaluation_time_seconds': time.time() - start_time, }

    datetime_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_units_str = "_".join(map(str, config.test_units)) if config.test_units else "all"
    ds_str = "_".join(config.n_ds).replace("-", "_")
    # Use eval_subsample_size in the filename
    subsample_str = f"_sub{config.eval_subsample_size}" if config.eval_subsample_size else ""
    results_filename = (f"{config.dataset}_{ds_str}_units_{test_units_str}_"
                        f"{config.policy_type}_B{config.budget}_S{seed}" # Use the current run's seed
                        f"{subsample_str}_" # Include subsample info
                        f"{datetime_stamp}_{config.results_file_suffix}")
    results_path = os.path.join(config.output_dir, results_filename)

    try: os.makedirs(os.path.dirname(results_path), exist_ok=True)
    except OSError as e: logging.error(f"Error creating output dir {os.path.dirname(results_path)}: {e}. Saving locally."); results_path = results_filename

    logging.info(f"Attempting to save results to: {results_path}")

    try:
        with open(results_path, 'w') as f: json.dump(results, f, indent=4, cls=NpEncoder)
        logging.info(f"Results successfully saved to {results_path}")
    except Exception as e: # Catch broader exceptions during saving
        logging.error(f"Error saving results to JSON '{results_path}': {e}. Check data types.", exc_info=True)
        print("\n--- Intervention Results (JSON save failed, printing summary) ---")
        try: # Fallback print
            simple_config = {k: v for k, v in serializable_config.items() if not isinstance(v, (list, dict)) or len(str(v)) < 100}
            print("Config Summary:", simple_config)
            print("\nMetrics History (First/Last Steps):")
            if intervention_metrics_history: print(f" Step 0: {intervention_metrics_history[0]}")
            if len(intervention_metrics_history) > 1: print(f" Step {len(intervention_metrics_history)-1}: {intervention_metrics_history[-1]}")
            print(f" Evaluated {n_samples_evaluated} samples.")
            print("-------------------------------------------------------------\n")
        except Exception as print_e: logging.error(f"Error during fallback print: {print_e}")

    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

if __name__ == '__main__':
    num_runs = 20
    original_argv = sys.argv
    try:
        FLAGS(original_argv, known_only=True)
    except flags.Error as e:
        print(f"FATAL Flags error: {e}", file=sys.stderr)
        sys.exit(1)

    base_seed = 0
    print(f"Starting {num_runs} evaluation runs with base seed {base_seed}...")
    print(f"Other configuration flags: {FLAGS.flag_values_dict()}")

    for i in range(num_runs):
        current_run_number = i + 1
        current_seed = base_seed + i
        print(f"\n===== Starting Run {current_run_number}/{num_runs} with Seed {current_seed} =====\n")
        try:
             # Call main with None for argv, it's handled inside now
             main(None, run_seed=current_seed)
        except Exception as e: # Catch errors per run
            # Log error with context
            logging.error(f"[Run {current_run_number}/{num_runs} Seed {current_seed}] Error occurred: {e}", exc_info=True)
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n===== Completed {num_runs} runs. =====\n")
    logging.shutdown()