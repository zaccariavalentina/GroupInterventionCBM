
import torch 
import torch.nn as nn
import torch.optim as optim
from absl import flags
import logging
from typing import Optional, Tuple, Dict
import torchmetrics

import enum_utils

# -- Helper functions -- 

def get_optimizer(model: nn.Module, config: flags.FlagValues) -> optim.Optimizer:
    """Creates optimizer based on flags"""
    lr = config.lr 
    wd = config.wd
    opt_name = config.optimizer.lower()

    if opt_name == 'sgd': 
        # logging.info(f"Using SGD optimizer with lr={lr}, momentum=0.9, weight_decay={wd}")
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif opt_name == 'adam':
        # logging.info(f"Using Adam optimizer with lr={lr}, weight_decay={wd} (Note: Adam's weight decay is L2 penalty)")
        # Adam's weight decay is not the same as AdamW's
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'adamw': 
        # logging.info(f"Using AdamW optimizer with lr={lr}, weight_decay={wd}")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def get_loss_functions_and_weights(arch: enum_utils.Arch, n_classes:int, device:torch.device, loss_weights_flag:Optional[list]) -> Tuple[list, list, list]:
    """Sets up loss functions and weights based on architecture."""
    loss_fns = []
    loss_names = []
    final_loss_weights = []

    # Default weights if not provided
    default_weights = {'concept': 1.0, 'class': 1.0}
    if loss_weights_flag:
        try:
            parsed_weights = [float(w) for w in loss_weights_flag]
            if arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_Y]:
                if len(parsed_weights) == 1:
                     if arch == enum_utils.Arch.X_TO_C: default_weights['concept'] = parsed_weights[0]
                     else: default_weights['class'] = parsed_weights[0]
                else:
                    logging.warning(f"Incorrect number of loss weights for {arch}. Expected 1, got {len(parsed_weights)}. Using defaults.")
            elif arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y]:
                 if len(parsed_weights) == 1 and arch == enum_utils.Arch.C_TO_Y:
                     default_weights['class'] = parsed_weights[0]
                 elif len(parsed_weights) == 2 and arch == enum_utils.Arch.X_TO_C_TO_Y:
                     default_weights['concept'] = parsed_weights[0]
                     default_weights['class'] = parsed_weights[1]
                 else:
                    logging.warning(f"Incorrect number of loss weights for {arch}. Expected {'1' if arch==enum_utils.Arch.C_TO_Y else '2'}, got {len(parsed_weights)}. Using defaults.")

        except ValueError:
            logging.warning(f"Invalid loss weights format: {loss_weights_flag}. Using defaults.")

    # Define losses based on architecture
    # BCEWithLogitsLoss is numerically stable and expects raw logits
    bce_loss = nn.BCEWithLogitsLoss().to(device)

    if arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_C_TO_Y]:
        loss_fns.append(bce_loss) # For multi-label concepts
        loss_names.append('concept_loss')
        final_loss_weights.append(default_weights['concept'])

    if arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y, enum_utils.Arch.X_TO_Y]:
        if n_classes == 1:
            loss_fns.append(bce_loss) # Binary classification
        else:
            # Assumes n_classes > 1 is multi-class single-label
            loss_fns.append(nn.CrossEntropyLoss().to(device)) # Expects (N, C) logits and (N,) target indices
        loss_names.append('class_loss')
        final_loss_weights.append(default_weights['class'])

    # logging.info(f"Using loss weights: {final_loss_weights} for losses: {loss_names}")
    return loss_fns, loss_names, final_loss_weights


def get_metrics(arch: enum_utils.Arch, n_concepts:int, n_classes:int, device:torch.device
    ) -> Tuple[Dict[str, torchmetrics.Metric], Dict[str, torchmetrics.Metric]]:
    """Intializes metrics using torchmetrics."""
    train_metrics = {}
    val_metrics = {}

    # common args for metrics 
    common_args_binary = {'task': 'binary'}
    common_args_multilabel = {'task': 'multilabel', 'num_labels': n_concepts}

    # -- Concept Metrics (if applicable) --
    if arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_C_TO_Y]: 
        # use multilabel AUCROC and Accuracy for concepts 
        # (concept loss is already logged in train function)
        train_metrics['concept_auroc'] = torchmetrics.AUROC(**common_args_multilabel).to(device)
        train_metrics['concept_accuracy'] = torchmetrics.Accuracy(**common_args_multilabel).to(device)
        val_metrics['concept_auroc'] = torchmetrics.AUROC(**common_args_multilabel).to(device)
        val_metrics['concept_accuracy'] = torchmetrics.Accuracy(**common_args_multilabel).to(device)

    # -- Class Metrics (if applicable) --
    if arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y, enum_utils.Arch.X_TO_Y]:
        if n_classes == 1: 
            # binary metrics 
            train_metrics['class_auroc'] = torchmetrics.AUROC(**common_args_binary).to(device)
            train_metrics['class_auprc'] = torchmetrics.AveragePrecision(**common_args_binary).to(device)
            train_metrics['class_accuracy'] = torchmetrics.Accuracy(**common_args_binary).to(device)
            val_metrics['class_auroc'] = torchmetrics.AUROC(**common_args_binary).to(device)
            val_metrics['class_auprc'] = torchmetrics.AveragePrecision(**common_args_binary).to(device) # AUPRC
            val_metrics['class_accuracy'] = torchmetrics.Accuracy(**common_args_binary).to(device)
        else: 
            # multi-class metrics (single label)
            common_args_multiclass = {'task': 'multiclass', 'num_classes': n_classes}
            train_metrics['class_accuracy_top1'] = torchmetrics.Accuracy(**common_args_multiclass, top_k=1).to(device)
            train_metrics['class_accuracy_top2'] = torchmetrics.Accuracy(**common_args_multiclass, top_k=2).to(device)
            train_metrics['class_accuracy_top3'] = torchmetrics.Accuracy(**common_args_multiclass, top_k=3).to(device)
            val_metrics['class_accuracy_top1'] = torchmetrics.Accuracy(**common_args_multiclass, top_k=1).to(device)
            val_metrics['class_accuracy_top2'] = torchmetrics.Accuracy(**common_args_multiclass, top_k=2).to(device)
            val_metrics['class_accuracy_top3'] = torchmetrics.Accuracy(**common_args_multiclass, top_k=3).to(device)

    return train_metrics, val_metrics

def update_metrics(metrics:Dict[str, torchmetrics.Metric], predictions:tuple, targets:tuple, arch:enum_utils.Arch, n_classes:int):
    """Updates metrics based on predictions and targets.
    
    Args: 
        metrics (dict): dictionary of metrics to update.
        targets (tuple): true concepts and/or class labels, (concept_labels, class_labels),
                         (concept_labels, ), or (class_labels, )
        predictions (tuple): similar to targets, but for model predictions.
        arch (enum_utils.Arch): architecture type.
        n_classes (int): number of classes.
    """
    pred_idx = 0
    target_idx = 0 
    
    if arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_C_TO_Y]:
        concept_preds = predictions[pred_idx]
        concept_targets = targets[target_idx].int() # accuracy expects int in targets
        # AUCROC expects probabilities or logits, accuracy needs specific format
        metrics['concept_auroc'].update(concept_preds, concept_targets)
        metrics['concept_accuracy'].update(torch.sigmoid(concept_preds), concept_targets) # use sigmoid for thresholding
        pred_idx += 1
        target_idx += 1
    
    if arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y, enum_utils.Arch.X_TO_Y]:
        class_preds = predictions[pred_idx]
        class_targets = targets[target_idx] # shape (N,1) or (N,)

        if n_classes == 1: 
            class_targets = class_targets.int()
            metrics['class_auroc'].update(class_preds, class_targets)
            metrics['class_auprc'].update(class_preds, class_targets)
            metrics['class_accuracy'].update(class_preds, class_targets)
        else: 
            class_targets = class_targets.squeeze().long() # cross entropy expects (N,), accuracy expects (N,) long
            metrics['class_accuracy_top1'].update(class_preds, class_targets)
            metrics['class_accuracy_top2'].update(class_preds, class_targets)
            metrics['class_accuracy_top3'].update(class_preds, class_targets)
    

def compute_metrics(metrics: Dict[str, torchmetrics.Metric]) -> Dict[str, float]:
    """Computes final values for all metrics."""
    results = {name: metric.compute().item() for name, metric in metrics.items()}
    return results

def reset_metrics(metrics: Dict[str, torchmetrics.Metric]):
    """Resets all metrics."""
    for metric in metrics.values():
        metric.reset()