import torch
import os
import shutil
import logging

logging.basicConfig(level=logging.WARNING)

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """Saves model checkpoint."""
    filepath = os.path.join(checkpoint_dir, filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)
        logging.info(f"Saved new best model to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")

    logging.info(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
         from collections import OrderedDict
         new_state_dict = OrderedDict()
         for k, v in state_dict.items():
             name = k[7:] if k.startswith('module.') else k
             new_state_dict[name] = v
         model.load_state_dict(new_state_dict)
    else:
         model.load_state_dict(state_dict)


    start_epoch = checkpoint['epoch']
    best_metric = checkpoint.get('best_metric', None) 

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            logging.warning(f"Could not load optimizer state, possibly due to parameter mismatch: {e}. Starting optimizer from scratch.")


    logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, best_metric: {best_metric})")
    return start_epoch, best_metric