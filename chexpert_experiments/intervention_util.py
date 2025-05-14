import logging
from typing import Dict, Tuple, Callable
from functools import partial
import enum

import torch
import torchmetrics
from torch.distributions import Bernoulli, kl

import enum_utils

class PolicyType(str, enum.Enum):
    RANDOM_CONCEPTS: str = 'random_concepts'
    RANDOM_GROUPS: str = 'random_groups'
    GREEDY_CONCEPTS: str = 'greedy_concepts'
    GREEDY_GROUPS: str = 'greedy_groups'
    OPTIMIZED: str = 'optimized'

def custom_update_metrics(metrics: Dict[str, torchmetrics.Metric],
        preds_tuple: Tuple[torch.Tensor, ...],
        targets_tuple: Tuple[torch.Tensor, ...],
        n_classes: int):
    """
    Custom metric update function for intervention evaluation step for binary classification.
    Handles shape and type requirements for binary classification metrics.

    Args:
        metrics (dict): Dictionary of class metrics to update (e.g., class_auroc).
        preds_tuple (tuple): Tuple containing class predictions (logits).
                             Expected: (class_preds_logits,)
        targets_tuple (tuple): Tuple containing class targets.
                               Expected: (class_targets,)
        n_classes (int): Number of classes (should be 1 for this use case).
    """
    if not preds_tuple or not targets_tuple:
        logging.warning("Empty predictions or targets tuple passed to custom_update_metrics.")
        return

    class_preds_logits = preds_tuple[0]
    class_targets = targets_tuple[0]

    if n_classes == 1:
        # Binary Classification
        # 1. Ensure Targets are Integer/Long and Shape (N,)
        if class_targets.dtype not in [torch.int, torch.long, torch.bool]:
            class_targets = class_targets.long() # Convert to long
        if class_targets.ndim == 2 and class_targets.shape[1] == 1:
            class_targets = class_targets.squeeze(-1) # Shape (N,)
        elif class_targets.ndim != 1:
            logging.warning(f"Unexpected class_targets shape in custom_update_metrics: {class_targets.shape}. Attempting flatten.")
            class_targets = class_targets.flatten()

        # 2. Ensure Predictions are Shape (N,)
        if class_preds_logits.ndim == 2 and class_preds_logits.shape[1] == 1:
            class_preds_logits = class_preds_logits.squeeze(-1) # Shape (N,)
        elif class_preds_logits.ndim != 1:
            logging.warning(f"Unexpected class_preds_logits shape in custom_update_metrics: {class_preds_logits.shape}. Attempting flatten.")
            class_preds_logits = class_preds_logits.flatten()

        # 3. Final Shape Check
        if class_preds_logits.shape != class_targets.shape:
            logging.error(f"Shape mismatch after adjustment in custom_update_metrics: preds {class_preds_logits.shape}, targets {class_targets.shape}")
            return # Skip update for this batch

        # 4. Calculate Probabilities for AUROC/AUPRC
        class_probs = torch.sigmoid(class_preds_logits) # Shape (N,)

        # 5. Update Metrics
        if 'class_auroc' in metrics:
            try:
                metrics['class_auroc'].update(class_probs, class_targets) # Needs probs (N,), target (N,) int
            except Exception as e:
                logging.error(f"Error updating class_auroc: {e}", exc_info=True)
        if 'class_auprc' in metrics:
            try:
                metrics['class_auprc'].update(class_probs, class_targets) # Needs probs (N,), target (N,) int
            except Exception as e:
                logging.error(f"Error updating class_auprc: {e}", exc_info=True)
        if 'class_accuracy' in metrics:
            # BinaryAccuracy can usually handle logits directly
            try:
                metrics['class_accuracy'].update(class_preds_logits, class_targets) # Logits (N,), Targets (N,) int
            except Exception as e:
                logging.error(f"Error updating class_accuracy: {e}", exc_info=True)
    else:
        raise ValueError(f"Unsupported number of classes for binary classification: {n_classes}. Expected 1.")


class CustomPolicyMetrics:
    """Collection of metrics for concept selection policies (to assign values to concepts)"""
    @staticmethod
    def concept_entropy(pred_concepts_logits: torch.Tensor) -> torch.Tensor:
        """
        Concept uncertainty: entropy of predicted concept distributions.
        Assumes independent Bernoulli distribution for each concept.
        Input Shape: (N, n_concepts) - Logits for each concept.
        Output Shape: (N, n_concepts) - Entropy for each concept prediction.
        """
        return Bernoulli(logits=pred_concepts_logits).entropy()

    @staticmethod
    def concept_confidence(pred_concepts_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence (probability of the most likely class) for concept predictions.
        Input Shape: (N, n_concepts) - Logits for each concept.
        Output Shape: (N, n_concepts) - Confidence for each concept prediction.
        """
        concept_probs = torch.sigmoid(pred_concepts_logits)
        # Confidence is max(p, 1-p)
        return torch.max(concept_probs, 1.0 - concept_probs)

    @staticmethod
    def _label_metric_base(concept_logits: torch.Tensor,
                           current_label_logits: torch.Tensor,
                           label_logits_if_0: torch.Tensor,
                           label_logits_if_1: torch.Tensor,
                           metric_type: str,
                           signed: bool) -> torch.Tensor:
        """
        Base calculation for label-based importance metrics assuming binary classification.

        Args:
            concept_logits (torch.Tensor): Logits of original concept predictions. Shape (N, n_concepts).
            current_label_logits (torch.Tensor): Logits of current label prediction. Shape (N,) or (N, 1).
            label_logits_if_0 (torch.Tensor): Logits of label prediction if concept j was 0. Shape (N, n_concepts).
            label_logits_if_1 (torch.Tensor): Logits of label prediction if concept j was 1. Shape (N, n_concepts).
            metric_type (str): 'entropy' or 'confidence'.
            signed (bool): If True, measure increase/decrease; otherwise, measure absolute change.

        Returns:
            torch.Tensor: Importance score for each concept. Shape (N, n_concepts).
        """
        if current_label_logits.ndim == 2 and current_label_logits.shape[1] == 1:
             current_label_logits = current_label_logits.squeeze(-1) # Shape (N,)
        if label_logits_if_0.ndim == 3 and label_logits_if_0.shape[-1] == 1:
             label_logits_if_0 = label_logits_if_0.squeeze(-1) # Shape (N, n_concepts)
        if label_logits_if_1.ndim == 3 and label_logits_if_1.shape[-1] == 1:
             label_logits_if_1 = label_logits_if_1.squeeze(-1) # Shape (N, n_concepts)

        # Ensure shapes are compatible for broadcasting if needed (though they should be correct now)
        assert current_label_logits.shape[0] == label_logits_if_0.shape[0] == label_logits_if_1.shape[0] == concept_logits.shape[0]
        assert label_logits_if_0.shape[1] == label_logits_if_1.shape[1] == concept_logits.shape[1]
        assert current_label_logits.ndim == 1 # Should be (N,)

        concept_probs = torch.sigmoid(concept_logits) # Shape (N, n_concepts)

        # Expand current label prediction to match the (N, n_concepts) shape for comparison
        current_label_logits_expanded = current_label_logits.unsqueeze(1).expand(-1, concept_logits.shape[1]) # Shape (N, n_concepts)

        if metric_type == 'entropy':
            current_metric_val = Bernoulli(logits=current_label_logits_expanded).entropy() # Shape (N, n_concepts)
            metric_val_if_0 = Bernoulli(logits=label_logits_if_0).entropy() # Shape (N, n_concepts)
            metric_val_if_1 = Bernoulli(logits=label_logits_if_1).entropy() # Shape (N, n_concepts)
            # Change = Original - New (so positive means entropy decrease)
            change_0 = current_metric_val - metric_val_if_0
            change_1 = current_metric_val - metric_val_if_1
        elif metric_type == 'confidence':
            # Confidence in the original predicted class
            current_dist = Bernoulli(logits=current_label_logits_expanded)
            pred_class = (current_label_logits_expanded > 0).float() # Most likely class based on original logits
            current_metric_val = torch.exp(current_dist.log_prob(pred_class)) # Use log_prob for stability, Shape (N, n_concepts)

            dist_0 = Bernoulli(logits=label_logits_if_0)
            dist_1 = Bernoulli(logits=label_logits_if_1)
            metric_val_if_0 = torch.exp(dist_0.log_prob(pred_class)) # Prob of original class if concept=0, Shape (N, n_concepts)
            metric_val_if_1 = torch.exp(dist_1.log_prob(pred_class)) # Prob of original class if concept=1, Shape (N, n_concepts)
            # Change = New - Original (so positive means confidence increase)
            change_0 = metric_val_if_0 - current_metric_val
            change_1 = metric_val_if_1 - current_metric_val
        else:
            raise ValueError(f"Unsupported metric_type: {metric_type}")

        if signed:
            # Expected change: p(c=1)*change_if_1 + p(c=0)*change_if_0
            metric = concept_probs * change_1 + (1.0 - concept_probs) * change_0
        else:
            # Expected absolute change
            metric = concept_probs * torch.abs(change_1) + (1.0 - concept_probs) * torch.abs(change_0)

        return metric # Shape (N, n_concepts)

    @staticmethod
    def label_entropy_change(concept_logits: torch.Tensor,
                             current_label_logits: torch.Tensor,
                             label_logits_if_0: torch.Tensor,
                             label_logits_if_1: torch.Tensor,
                             signed: bool):
        """Measure importance via expected change in label entropy. See _label_metric_base."""
        return CustomPolicyMetrics._label_metric_base(
            concept_logits, current_label_logits, label_logits_if_0, label_logits_if_1, 'entropy', signed)

    @staticmethod
    def label_confidence_change(concept_logits: torch.Tensor,
                                current_label_logits: torch.Tensor,
                                label_logits_if_0: torch.Tensor,
                                label_logits_if_1: torch.Tensor,
                                signed: bool):
        """Measure importance via expected change in label confidence. See _label_metric_base."""
        return CustomPolicyMetrics._label_metric_base(
            concept_logits, current_label_logits, label_logits_if_0, label_logits_if_1, 'confidence', signed)

    @staticmethod
    def label_kld(concept_logits: torch.Tensor,
                  current_label_logits: torch.Tensor,
                  label_logits_if_0: torch.Tensor,
                  label_logits_if_1: torch.Tensor) -> torch.Tensor:
        """
        Measure importance via expected symmetric KL divergence between original and intervened label distributions.
        Assumes binary classification.

        Args:
            concept_logits (torch.Tensor): Logits of original concept predictions. Shape (N, n_concepts).
            current_label_logits (torch.Tensor): Logits of current label prediction. Shape (N,) or (N, 1).
            label_logits_if_0 (torch.Tensor): Logits of label prediction if concept j was 0. Shape (N, n_concepts).
            label_logits_if_1 (torch.Tensor): Logits of label prediction if concept j was 1. Shape (N, n_concepts).

        Returns:
            torch.Tensor: Importance score for each concept. Shape (N, n_concepts).
        """
        if current_label_logits.ndim == 2 and current_label_logits.shape[1] == 1:
             current_label_logits = current_label_logits.squeeze(-1) # Shape (N,)
        if label_logits_if_0.ndim == 3 and label_logits_if_0.shape[-1] == 1:
             label_logits_if_0 = label_logits_if_0.squeeze(-1) # Shape (N, n_concepts)
        if label_logits_if_1.ndim == 3 and label_logits_if_1.shape[-1] == 1:
             label_logits_if_1 = label_logits_if_1.squeeze(-1) # Shape (N, n_concepts)

        # Ensure shapes are compatible
        assert current_label_logits.shape[0] == label_logits_if_0.shape[0] == label_logits_if_1.shape[0] == concept_logits.shape[0]
        assert label_logits_if_0.shape[1] == label_logits_if_1.shape[1] == concept_logits.shape[1]
        assert current_label_logits.ndim == 1 # Should be (N,)

        concept_probs = torch.sigmoid(concept_logits) # Shape (N, n_concepts)

        # Expand current label prediction to match the (N, n_concepts) shape for KL calculation
        current_label_logits_expanded = current_label_logits.unsqueeze(1).expand(-1, concept_logits.shape[1]) # Shape (N, n_concepts)

        current_dist = Bernoulli(logits=current_label_logits_expanded)
        dist_0 = Bernoulli(logits=label_logits_if_0)
        dist_1 = Bernoulli(logits=label_logits_if_1)

        # Symmetric KL divergence: KL(P||Q) + KL(Q||P)
        # Note: kl.kl_divergence(bernoulli1, bernoulli2) computes KL for each element pair.
        kld0 = kl.kl_divergence(dist_0, current_dist) + kl.kl_divergence(current_dist, dist_0) # Shape (N, n_concepts)
        kld1 = kl.kl_divergence(dist_1, current_dist) + kl.kl_divergence(current_dist, dist_1) # Shape (N, n_concepts)

        # Expected KL divergence
        return concept_probs * kld1 + (1.0 - concept_probs) * kld0 # Shape (N, n_concepts)

# --- Utility Function ---
def get_metric_fn(metric: enum_utils.Metric) -> Callable:
    """Utility function to get metric functions"""
    if not isinstance(metric, enum_utils.Metric):
        try:
            metric = enum_utils.Metric(metric)
        except ValueError:
            raise ValueError(f"Invalid metric value: {metric}")

    if metric == enum_utils.Metric.CONCEPT_ENTROPY:
        return CustomPolicyMetrics.concept_entropy
    elif metric == enum_utils.Metric.CONCEPT_CONFIDENCE:
        return CustomPolicyMetrics.concept_confidence
    elif metric == enum_utils.Metric.LABEL_ENTROPY_CHANGE:
        return partial(CustomPolicyMetrics.label_entropy_change, signed=False)
    elif metric == enum_utils.Metric.LABEL_ENTROPY_DECREASE:
        return partial(CustomPolicyMetrics.label_entropy_change, signed=True)
    elif metric == enum_utils.Metric.LABEL_CONFIDENCE_CHANGE:
        return partial(CustomPolicyMetrics.label_confidence_change, signed=False)
    elif metric == enum_utils.Metric.LABEL_CONFIDENCE_INCREASE:
        return partial(CustomPolicyMetrics.label_confidence_change, signed=True)
    elif metric == enum_utils.Metric.LABEL_KLD:
        return CustomPolicyMetrics.label_kld
    else:
        raise ValueError(f'Metric {metric} not supported yet.')