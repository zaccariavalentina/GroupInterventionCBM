# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Adaptation Copyright 2024 Your Name or Affiliation
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

"""Data loading utilities for the CheXpert [1] dataset (PyTorch version).

[1] Jeremy Irvin*, Pranav Rajpurkar*, Michael Ko, Yifan Yu,
Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball,
Katie Shpanskaya, Jayne Seekins, David A. Mong, Safwan S. Halabi,
Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz,
Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. CheXpert: A Large Chest
Radiograph Dataset with Uncertainty Labels and Expert Comparison, AAAI 2019.
"""
import dataclasses
import functools
import os
import logging

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
from PIL import Image
import numpy as np

from typing import Optional, Tuple, Callable, Dict, Any, List

from . import preproc_util

"""The final label is derived from 'No Finding' in the original dataset. 
We need to map this to a binary class label. Usually 1 if any pathology is 
present, 0 otherwise. 
"""

_CLASS_LABEL_COLUMN = 'No Finding'
_UNCERTAIN_VALUE = -1.0
_NEGATIVE_VALUE = 0.0
_POSITIVE_VALUE = 1.0

_PATHOLOGIES = (
    'enlarged_cardiom',
    'cardiomegaly',
    # The order of Lung Opacity and Lung Lesion is flipped in comparison to Table 1 in [1].
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_0ther',
    'fracture',
    'support_devices'
)

@dataclasses.dataclass(frozen=True, init=False)
class Config: 
    n_concepts = 13
    n_classes = 1 
    image_size = (320, 320, 3)
    data_dir = ('/mnt/disk1/valentina_zaccaria/smallchexpert') # Update with correct dir where the dataset is stored 
    train_csv = 'train.csv'
    val_csv = 'valid.csv'

@dataclasses.dataclass(frozen=True, init=False)
class CostSpec: 
    """Defines the acquisition costs for different concepts.

    "default" corresponds to acquisition costs for all concepts other than
    cardiomegaly, fracture and support_devices.
    """
    default = 10
    cardiomegaly = 3
    fracture = 1
    support_devices = 1

    @classmethod 
    def get_cost(cls, concept_group_name): 
        return getattr(cls, concept_group_name, cls.default)


def load_concept_groups(): 
    """Loads concept group information."""
    concept_groups = {}
    for concept_i in range(Config.n_concepts):
        concept_groups[_PATHOLOGIES[concept_i]] = [concept_i]
    return concept_groups

def load_concept_names(): 
    return _PATHOLOGIES

def load_hardcoded_detailed_costs() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Loads the HARDCODED detailed group cost structure provided by the user.

    Returns the structure in the format expected by the optimizer:
        group_name -> {
            'setup_cost': float,
            'concepts': { concept_name: marginal_cost (float) }
        }
    Also returns the list of concept names defined within this structure.
    """
    group_cost_structure_raw = {
        'Xray_radiologist': {
            'setup_cost': 6.0,
            'marginal_cost': {
                'enlarged_cardiom': 1.5, 'cardiomegaly': 1.5, 'lung_opacity': 2.5,
                'atelectasis': 2.3, 'pneumothorax': 2.0, 'fracture': 1.0, 
                'support_devices': 0.8, 
            }},

        'Xray_non_expert': {
            'setup_cost': 2.0, 
            'marginal_cost': {
                'fracture': 2.5,        
                'support_devices': 2.0, 
            }},
        
        'Xray_nurse': {
            'setup_cost': 3.0, 
            'marginal_cost': {
                'cardiomegaly': 2.5,    
                'fracture': 2.0,        
                'support_devices': 1.8, 
            }},
        
        'CT_scan_specialist': {
            'setup_cost': 10.0, 
            'marginal_cost': {
                'lung_lesion': 4.0, 'consolidation': 3.5, 'pneumonia': 4.0, 
                'pleural_effusion': 3.0, 'pleural_0ther': 3.0,
            }},

        'Ultrasound_specialist': {
            'setup_cost': 6.0,
            'marginal_cost': {
                'edema': 2.5,        
                'pneumothorax': 2.2, 
            }},
        
        'Lung_radiologist': {
            'setup_cost': 7.0,
            'marginal_cost': {
                'lung_lesion': 5.0, 
                'pneumonia': 3.8,   
                'edema': 3.2,       
                'consolidation': 3.0, 
            }},
    }

    # Convert to the format expected by the optimizer ('concepts' key)
    # And collect all unique concept names defined in this structure
    group_cost_structure_final = {}
    all_concepts_in_structure = set()
    for group_name, data in group_cost_structure_raw.items():
         marginal_costs = data.get('marginal_cost', {})
         group_cost_structure_final[group_name] = {
             'setup_cost': data.get('setup_cost', 0.0),
             'concepts': marginal_costs # Rename key to 'concepts'
         }
         all_concepts_in_structure.update(marginal_costs.keys())

    # Define the canonical list of concept names based SOLELY on this structure
    final_concept_names = sorted(list(all_concepts_in_structure))

    # Validate against _PATHOLOGIES (optional but recommended)
    original_pathologies_set = set(_PATHOLOGIES)
    if all_concepts_in_structure != original_pathologies_set:
         logging.warning("Concepts defined in hardcoded cost structure differ from _PATHOLOGIES.")
         logging.warning(f"Concepts in structure ONLY: {all_concepts_in_structure - original_pathologies_set}")
         logging.warning(f"Concepts in _PATHOLOGIES ONLY: {original_pathologies_set - all_concepts_in_structure}")
         
    return group_cost_structure_final, final_concept_names


def get_min_cost_per_concept():
    group_cost_structure, _ = load_hardcoded_detailed_costs()
    
    concept_min_costs = {}

    for group_name, group_data in group_cost_structure.items():
        setup_cost = group_data['setup_cost']
        for concept, marginal_cost in group_data['concepts'].items():
            total_cost = setup_cost + marginal_cost
            if concept not in concept_min_costs or total_cost < concept_min_costs[concept]['total_cost']:
                concept_min_costs[concept] = {
                    'group': group_name,
                    'setup_cost': setup_cost,
                    'marginal_cost': marginal_cost,
                    'total_cost': total_cost
                }
    return concept_min_costs


def load_concept_costs(concept_groups, **_): 
    """Loads concept label acquisiton costs. 
    
    We assign acquisition costs to concepts based on a crude estimation of
    annotation difficulty and the degree of annotator expertise required for this
    dataset. We consider
        "Fracture" and "Support Devices" concepts as easy to annotate as even a
        non-radiologist can identify these,
        "Cardiomegaly" as having medium annotation difficulty as a non-radiologist
        could measure for this using callipers and some heuristics about the
        required ratio, and,
        all the remaining concepts as hard to annotate owing to a lot of
        within-label variation requiring an expert radiologist's opinion.
    The assigned concept costs quantify these qualitative annotation difficult
    estimates.
    
    Args: 
        concept_groups (dict): dictionary containing concept group names as keys and
                               list of concept indices as values, as returned by 
                               load_concept_groups()
    
    Returns: 
        concept_costs (dict): dictionary mapping concept group names to their respective
                              label acquisition costs.
    """
    concept_costs = {}
    for concept_group_name in concept_groups:
        concept_costs[concept_group_name] = CostSpec.get_cost(concept_group_name)
    return concept_costs


class CheXpertDataset(Dataset): 
    """PyTorch Dataset for CheXpert"""

    def __init__(self, csv_path:str, image_root_dir:str, transform:Optional[Callable]=None): 
        """
        Args: 
            csv_path (string): Path to the csv file with annotations. 
                               Assumes columns: 'Path', _PATHOLOGIES..., _CLASS_LABEL_COLUMN
            image_root_dir (string): Directory with all the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.image_root_dir = image_root_dir
        
        # read csv, handle nan by mapping to 0, 
        self.annotations = pd.read_csv(csv_path).fillna(_NEGATIVE_VALUE)
        self.pathology_columns = self.annotations.columns[6:] 

        # ensure class label columns exists
        if _CLASS_LABEL_COLUMN not in self.annotations.columns:
            raise ValueError(f"Missing required class label column in CSV: {_CLASS_LABEL_COLUMN}")

        self.transform = transform 

        # precompute label mapping for efficiency 
        self.concept_map = {
            _UNCERTAIN_VALUE: 0.0, # uncertain -> 0 (binary label)
            _NEGATIVE_VALUE: 0.0,  # negative/unmentioned -> 0 
            _POSITIVE_VALUE: 1.0   # positive -> 1
        }
        self.uncertainty_map = {
             _UNCERTAIN_VALUE: 1.0, # Uncertain -> 1 (uncertainty flag)
             _NEGATIVE_VALUE: 0.0,  # Negative/Unmentioned -> 0
             _POSITIVE_VALUE: 0.0   # Positive -> 0
        }

    def __len__(self): 
        return len(self.annotations)
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        img_path_rel = self.annotations.iloc[idx]['Path'].replace('CheXpert-v1.0-small/', '')
        img_path_full = os.path.join(self.image_root_dir, img_path_rel) #TODO: check, we might adjust to strig the initial part 

        try: 
            # chexpert images are grayscale but InceptionV3 expects 3 channels. 
            # load as grayscale and convert to RGB by repeating the channel
            image = Image.open(img_path_full).convert('L').convert('RGB')
        except FileNotFoundError: 
            # raise FileNotFoundError(f"Image not found: {img_path_full}")
            return None
        except Exception as e: 
            # print(f"Error loading image {img_path_full}: {e}")
            # raise e
            return None
    
        # extract concept labels (pathologies)
        concept_values = self.annotations.iloc[idx][self.pathology_columns].values.astype(np.float32)

        # apply mappings 
        # ensure input values match keys in map before mapping 
        concept_labels = np.vectorize(self.concept_map.get)(concept_values).astype(np.float32)
        concept_uncertainty = np.vectorize(self.uncertainty_map.get)(concept_values).astype(np.float32)

        # extract class label ('No Finding') - map 1.0 (No Finding) to 0, others (finding present) to 1. 
        # treat uncertain as 1.0 (present)
        no_finding_label = self.annotations.iloc[idx][_CLASS_LABEL_COLUMN]
        class_label = 0.0 if no_finding_label == _POSITIVE_VALUE else 1.0
        class_label = np.array([class_label], dtype=np.float32)

        # apply transform 
        if self.transform: 
            image = self.transform(image)
        else: 
            # default minimal transform
            image = transforms.ToTensor()(image) # converts to (C, H, W) and scales to [0, 1]
        
        # convert labels to tensors 
        concept_labels = torch.from_numpy(concept_labels)
        concept_uncertainty = torch.from_numpy(concept_uncertainty)
        class_label = torch.from_numpy(class_label)

        return image, concept_labels, class_label, concept_uncertainty


def get_chexpert_transforms(train=True, image_size=(320, 320), brightness_delta=0.1):
    """Gets the appropriate Torchvision transforms."""
    # normalization for ImageNet pretrained models 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #TODO: check. weird. not in the original code

    if train: 
        def train_transform_fn(img): 
            # PIL to tensor
            img = transforms.ToTensor()(img)
            # center crop and resize 
            img = preproc_util.center_crop_and_resize(img, image_size[0], image_size[1], crop_proportion=1.0)
            # random horizontal flip
            if np.random.rand() > 0.5:
                img = transforms.functional.hflip(img)
            # random brightness 
            img = preproc_util.random_brightness(img, brightness_delta)
            # normalize 
            img = normalize(img) #TODO: check. weird. not in the original code
            return img
        return train_transform_fn
    else: 
        # deterministic eval transform
        return transforms.Compose([
            transforms.Lambda(lambda img: preproc_util.center_crop_and_resize(img, image_size[0], image_size[1], crop_proportion=1.0)),
            transforms.ToTensor(),
            normalize, #TODO: check. weird. not in the original code
        ])
    

def load_dataset(batch_size:int = 32, 
                 merge_train_and_val:bool = True, 
                 num_workers: int = 4
    )-> Tuple[DataLoader, DataLoader, Optional[DataLoader]]: 
    """Loads the CheXpert dataset using PyTorch DataLoaders.
    
    Args:
        batch_size (int): Batch size for the DataLoader.
        merge_train_and_val (bool): Whether to merge the training and validation datasets.
        num_workers (int): Number of worker threads for data loading.
    
    Returns:
        Tuple[DataLoader, DataLoader, Optional[DataLoader]]: Tuple of DataLoaders for
            training, validation, and test datasets. Test loader is None if merge_train_and_val is True.
    """
    datadir = Config.data_dir 
    train_csv_path = os.path.join(datadir, Config.train_csv)
    val_csv_path = os.path.join(datadir, Config.val_csv)
    image_root = datadir # assuming images are in the same directory as the csv files

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"CSV files not found in {datadir}")
    
    img_size = (Config.image_size[0], Config.image_size[1])
    train_transform = get_chexpert_transforms(train=True, image_size=img_size)
    eval_transform = get_chexpert_transforms(train=False, image_size=img_size)

    if merge_train_and_val: 
        print('Merging train and val datasets')
        train_ds_part = CheXpertDataset(
            csv_path=train_csv_path,
            image_root_dir=image_root,
            transform=train_transform
        )
        val_ds_part = CheXpertDataset(
            csv_path=val_csv_path,
            image_root_dir=image_root,
            transform=eval_transform
        )

        # filter out None Items potentially returned by __getitem__ on error
        train_ds_part_filtered = [item for item in train_ds_part if item is not None]
        val_ds_part_filtered = [item for item in val_ds_part if item is not None]

        if len(train_ds_part_filtered) != len(train_ds_part):
             print(f"Warning: Filtered {len(train_ds_part) - len(train_ds_part_filtered)} items from train due to loading errors.")
        if len(val_ds_part_filtered) != len(val_ds_part):
             print(f"Warning: Filtered {len(val_ds_part) - len(val_ds_part_filtered)} items from val due to loading errors.")

        # wrap filtered lists back into Dataset objects for ConcatDataset if using it 
        # Simpler: just use the filtered lists directly if ConcatDataset isn't strictly needed, but DataLoader prefers Dataset obj
        # Let's create Subset datasets from the original ones based on valid indices
        valid_train_indices = [i for i, item in enumerate(train_ds_part) if item is not None]
        valid_val_indices = [i for i, item in enumerate(val_ds_part) if item is not None]

        train_ds_subset = Subset(train_ds_part, valid_train_indices)
        val_ds_subset = Subset(val_ds_part, valid_val_indices)

        full_train_ds = ConcatDataset([train_ds_subset, val_ds_subset])

        # use the validation set defined by val_csv_patth as the final test test 
        val_ds = CheXpertDataset(csv_path=val_csv_path, image_root_dir=image_root, transform=eval_transform)
        valid_final_val_indices = [i for i, item in enumerate(val_ds) if item is not None]
        val_ds_filtered = Subset(val_ds, valid_final_val_indices)

        train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        # use the original validation split for validation during training
        val_loader = DataLoader(val_ds_filtered, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = None # No separate test set in this configuration

    else: 
        # print("Using train split for training, validation split for validation.")
        train_ds = CheXpertDataset(csv_path=train_csv_path, image_root_dir=image_root, transform=train_transform)
        val_ds = CheXpertDataset(csv_path=val_csv_path, image_root_dir=image_root, transform=eval_transform)

        train_ds_filtered = train_ds
        val_ds_filtered = val_ds

        train_loader = DataLoader(train_ds_filtered, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds_filtered, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = None # Assuming no separate test set file for now

    return train_loader, val_loader, test_loader