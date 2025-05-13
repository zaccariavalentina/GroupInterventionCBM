# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive bottleneck training and evaluation (PyTorch version)."""

import os

import time
import logging
from typing import Sequence, Dict, Any, Tuple, Optional

from absl import app
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics # Ensure torchmetrics is installed

import enum_utils
import network
from datasets import chexpert_dataset
import utils
import train_util


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


_ARCH = flags.DEFINE_enum_class(
    'arch',
    default=enum_utils.Arch.X_TO_C_TO_Y_SIGMOID,
    enum_class=enum_utils.Arch,
    help='Architecture to use for training.')
_NON_LINEAR_CTOY = flags.DEFINE_bool(
    'non_linear_ctoy',
    default=False,
    help='Whether to use a non-linear CtoY model.')
_DATASET = flags.DEFINE_enum_class(
    'dataset',
    default=enum_utils.Dataset.CHEXPERT,
    enum_class=enum_utils.Dataset,
    help='Dataset to use for training.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 
    default=32, 
    help='Batch Size')
_MERGE_TRAIN_AND_VAL = flags.DEFINE_bool(
    'merge_train_and_val',
    default=False,
    help='Whether to merge training and validation sets for training.')
_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    default='sgd',
    enum_values=['sgd', 'adam'],
    help='Optimizer to use for training.')
_LR = flags.DEFINE_float(
    'lr', 
    default=1e-3, 
    help='Learning rate.')
_WD = flags.DEFINE_float(
    'wd', 
    default=0, 
    help='Weight decay.')
_LOSS_WEIGHTS = flags.DEFINE_list(
    'loss_weights', 
    default=None, 
    help='Loss weights')
_EPOCHS = flags.DEFINE_integer(
    'epochs', 
    default=50, 
    help='No. of epochs.')
_STOPPING_PATIENCE = flags.DEFINE_integer(
    'stopping_patience',
    default=5,
    help='Patience (in no. of epochs) for early stopping.')
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default='CBM_results',
    help='Experiment directory to save models and results.')
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 
    default=4, 
    help='Number of DataLoader workers.')
_CHECKPOINT_TO_LOAD = flags.DEFINE_string(
    'load_checkpoint', 
    default=None, 
    help='Path to checkpoint file to load and resume training.')

_SEED = flags.DEFINE_integer(
    'seed', 
    default=None, 
    help='Random seed for reproducibility.')

FLAGS = flags.FLAGS

import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Train and Evaluate Functions ---
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader, 
                    optimizer: optim.Optimizer, 
                    loss_fns: list,
                    loss_weights: list,
                    metrics: Dict[str, torchmetrics.Metric],
                    device: torch.device,
                    epoch: int,
                    writer: SummaryWriter) -> Dict[str, float]:
    """Runs one training epoch."""
    model.train()
    train_util.reset_metrics(metrics)
    total_loss = 0.0
    running_losses = {f'loss_{i}': 0.0 for i in range(len(loss_fns))}

    start_time = time.time()
    for batch_idx, data in enumerate(dataloader):
        data = [d.to(device) for d in data] 
        inputs, targets = model.get_x_y_from_data(data)

        optimizer.zero_grad()
        predictions = model(inputs)

        # compute losses: the batch loss is the weighted sum of all losses  
        batch_loss = 0.0
        for i, (loss_fn, weights, pred, target) in enumerate(zip(loss_fns, loss_weights, predictions, targets)):
            # ensure target is float for BCE loss
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                target = target.float()
            # ensure target is long and squeezed for CrossEntropy
            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                target = target.squeeze().long()
            
            loss_val = loss_fn(pred, target)
            running_losses[f'loss_{i}'] += loss_val.item()
            batch_loss += weights * loss_val
    
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() 
        train_util.update_metrics(metrics, predictions, targets, model.arch, model.n_classes)

        if batch_idx % 100 == 0: # Log progress periodically
            logging.info(f'Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Batch Loss: {batch_loss.item():.4f}')

        epoch_duration = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        epoch_metrics = train_util.compute_metrics(metrics)

    # Log metrics and loss to console and TensorBoard
    logging.info(f'Epoch {epoch} Train | Avg Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s')
    log_str = " | ".join([f"{name}: {val:.4f}" for name, val in epoch_metrics.items()])
    logging.info(f'Epoch {epoch} Train Metrics | {log_str}')       


    writer.add_scalar('Loss/Train', avg_loss, epoch)
    for name, value in epoch_metrics.items():
        writer.add_scalar(f'Metrics/Train/{name}', value, epoch)
    # Log individual running losses
    for i, loss_fn_name in enumerate(running_losses.keys()):
        avg_running_loss = running_losses[loss_fn_name] / len(dataloader)
        writer.add_scalar(f'Loss/Train/{loss_fn_name}', avg_running_loss, epoch)

    return {'loss': avg_loss, **epoch_metrics}

def evaluate(model: nn.Module,
             dataloader: DataLoader,
             loss_fns: list,
             loss_weights: list,
             metrics: Dict[str, torchmetrics.Metric],
             device: torch.device,
             epoch: int,
             writer: SummaryWriter,
             prefix: str = 'Val') -> Dict[str, float]:
    """Runs evaluation on the validation or test set."""
    model.eval()
    train_util.reset_metrics(metrics)
    total_loss = 0.0
    running_losses = {f'loss_{i}': 0.0 for i in range(len(loss_fns))}

    start_time = time.time()
    with torch.no_grad(): # Disable gradient calculation
        for data in dataloader:
            data = [d.to(device) for d in data]
            inputs, targets = model.get_x_y_from_data(data)

            predictions = model(inputs)

            # Calculate loss
            batch_loss = 0.0
            for i, (loss_fn, weight, pred, target) in enumerate(zip(loss_fns, loss_weights, predictions, targets)):
                 if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    target = target.float()
                 elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    target = target.squeeze().long()

                 loss_val = loss_fn(pred, target)
                 running_losses[f'loss_{i}'] += loss_val.item()
                 batch_loss += weight * loss_val

            total_loss += batch_loss.item()
            train_util.update_metrics(metrics, predictions, targets, model.arch, model.n_classes)

    eval_duration = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    eval_metrics = train_util.compute_metrics(metrics)

    # Log metrics and loss
    logging.info(f'Epoch {epoch} {prefix} | Avg Loss: {avg_loss:.4f} | Duration: {eval_duration:.2f}s')
    log_str = " | ".join([f"{name}: {val:.4f}" for name, val in eval_metrics.items()])
    logging.info(f'Epoch {epoch} {prefix} Metrics | {log_str}')

    if writer:
        writer.add_scalar(f'Loss/{prefix}', avg_loss, epoch)
        for name, value in eval_metrics.items():
            writer.add_scalar(f'Metrics/{prefix}/{name}', value, epoch)
        # Log individual running losses
        for i, loss_fn_name in enumerate(running_losses.keys()):
            avg_running_loss = running_losses[loss_fn_name] / len(dataloader)
            writer.add_scalar(f'Loss/{prefix}/{loss_fn_name}', avg_running_loss, epoch)

    return {'loss': avg_loss, **eval_metrics}


# -- Main Function -- 
def main(argv: Sequence[str]):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    config = FLAGS
    if config.seed: 
        set_seed(config.seed)
    logging.info(f"Random seed set to {config.seed}")

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    if config.dataset == enum_utils.Dataset.CHEXPERT:
        dataset_module = chexpert_dataset
    else:
        raise ValueError('Dataset not supported.')

    # Create experiment directory structure
    base_checkpoint_dir = os.path.join(
        config.experiment_dir, config.dataset.value, config.arch.value,
        f'{config.optimizer}_lr-{config.lr}_wd-{config.wd}'
    )
    log_dir = os.path.join(
        config.experiment_dir, config.dataset.value, 'logs', config.arch.value,
        f'{config.optimizer}_lr-{config.lr}_wd-{config.wd}'
    )
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"Checkpoints will be saved to: {base_checkpoint_dir}")
    logging.info(f"TensorBoard logs will be saved to: {log_dir}")


    # --- Load Data ---
    logging.info("Loading datasets...")
    ds_train, ds_val, ds_test = dataset_module.load_dataset(
        batch_size=config.batch_size,
        merge_train_and_val=config.merge_train_and_val,
        num_workers=config.num_workers
    )
    logging.info(f"Train batches: {len(ds_train)}, Val batches: {len(ds_val)}, Test batches: {len(ds_test) if ds_test else 'N/A'}")


    # --- Create Model ---
    logging.info(f"Creating model: {config.arch.value}")
    model = network.InteractiveBottleneckModel(
        arch=config.arch,
        n_concepts=dataset_module.Config.n_concepts,
        n_classes=dataset_module.Config.n_classes,
        non_linear_ctoy=config.non_linear_ctoy,
    ).to(device)
    logging.info(f'Model initialized: {model.arch}, n_concepts={model.n_concepts}, n_classes={model.n_classes}')
    # logging.info(model) # Print model structure if needed

    print(model.arch)
    print('not expected')


    # --- Setup Optimizer, Loss, Metrics ---
    optimizer = train_util.get_optimizer(model, config)
    loss_fns, loss_names, loss_weights = train_util.get_loss_functions_and_weights(
        model.arch, model.n_classes, device, config.loss_weights)
    train_metrics, val_metrics = train_util.get_metrics(
        model.arch, model.n_concepts, model.n_classes, device)


    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_metric = -float('inf') # Assume higher is better for primary metric
    epochs_no_improve = 0

    if config.load_checkpoint:
        try:
            start_epoch, loaded_best_metric = utils.load_checkpoint(
                config.load_checkpoint, model, optimizer
            )
            if loaded_best_metric is not None:
                best_val_metric = loaded_best_metric
            logging.info(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            logging.warning(f"Checkpoint file not found at {config.load_checkpoint}. Starting training from scratch.")
        except Exception as e:
             logging.error(f"Error loading checkpoint: {e}. Starting training from scratch.")


    # --- Select Metric to Monitor for Best Model/Early Stopping ---
    # Prioritize class AUROC/Accuracy if available, else concept AUROC/Accuracy, else loss
    monitor_metric_name = None
    mode = 'max' # Default: higher is better

    if model.arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y, enum_utils.Arch.X_TO_Y]:
        if model.n_classes == 1:
            monitor_metric_name = 'class_auroc' # Primary metric
        else:
             monitor_metric_name = 'class_accuracy_top1' # Primary metric
    elif model.arch == enum_utils.Arch.X_TO_C:
        monitor_metric_name = 'concept_auroc'

    if monitor_metric_name is None or monitor_metric_name not in val_metrics:
         monitor_metric_name = 'loss' # Fallback to loss
         mode = 'min' # Lower loss is better
         best_val_metric = float('inf')

    logging.info(f"Monitoring validation metric: '{monitor_metric_name}' (mode: {mode}) for best model and early stopping.")


    # --- Training Loop ---
    logging.info("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        logging.info(f"--- Epoch {epoch}/{config.epochs-1} ---")

        train_results = train_one_epoch(
            model, ds_train, optimizer, loss_fns, loss_weights, train_metrics, device, epoch, writer
        )

        val_results = evaluate(
            model, ds_val, loss_fns, loss_weights, val_metrics, device, epoch, writer, prefix='Val'
        )

        # --- Checkpointing and Early Stopping ---
        current_val_metric = val_results.get(monitor_metric_name, None)
        if current_val_metric is None:
             logging.warning(f"Monitor metric '{monitor_metric_name}' not found in validation results. Cannot perform checkpointing/early stopping.")
             continue # Skip saving/stopping logic if metric is missing

        is_best = False
        if mode == 'max' and current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            is_best = True
            epochs_no_improve = 0
        elif mode == 'min' and current_val_metric < best_val_metric:
            best_val_metric = current_val_metric
            is_best = True
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save checkpoint (last and best)
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': best_val_metric,
            'monitor_metric': monitor_metric_name,
            'arch': model.arch,
            'n_concepts': model.n_concepts,
            'n_classes': model.n_classes,
        }
        utils.save_checkpoint(checkpoint_state, is_best, base_checkpoint_dir,
                              filename='checkpoint_last.pth.tar',
                              best_filename=f'checkpoint_best_{monitor_metric_name}.pth.tar')

        # Check for early stopping
        if epochs_no_improve >= config.stopping_patience:
            logging.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement on {monitor_metric_name}.")
            break

    writer.close()
    logging.info("Training finished.")

    # --- Final Evaluation (Optional) ---
    logging.info("Loading best model for final evaluation...")
    best_model_path = os.path.join(base_checkpoint_dir, f'checkpoint_best_{monitor_metric_name}.pth.tar')
    if os.path.exists(best_model_path):
        try:
            _, _ = utils.load_checkpoint(best_model_path, model) # Load best weights, ignore epoch/optimizer
            logging.info(f"Loaded best model from {best_model_path} based on '{monitor_metric_name}'")

            # Evaluate on validation set
            logging.info("Evaluating best model on validation set...")
            final_val_results = evaluate(
                model, ds_val, loss_fns, loss_weights, val_metrics, device, config.epochs, writer=None, prefix='FinalVal'
            )

            # Evaluate on test set if available
            if ds_test:
                logging.info("Evaluating best model on test set...")
                # Ensure test metrics are initialized if needed (can reuse val_metrics instances)
                test_metrics = val_metrics # Reuse metric objects, they will be reset
                final_test_results = evaluate(
                    model, ds_test, loss_fns, loss_weights, test_metrics, device, config.epochs, writer=None, prefix='FinalTest'
                )
            else:
                logging.info("No test set provided for final evaluation.")

        except Exception as e:
             logging.error(f"Error during final evaluation: {e}")

    else:
        logging.warning(f"Best model checkpoint not found at {best_model_path}. Skipping final evaluation.")


if __name__ == '__main__':
    app.run(main)