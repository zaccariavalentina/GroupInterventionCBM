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

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import enum_utils
from typing import Tuple

class InteractiveBottleneckModel(nn.Module): 
    """Interactive Bottleneck Model class (PyTorch)"""
    def __init__(self, 
                 n_concepts: int, 
                 n_classes: int, 
                 arch: enum_utils.Arch = enum_utils.Arch.X_TO_C_TO_Y, 
                 non_linear_ctoy: bool = False):
        """Initializes an InteractiveBottleneckModel instance.
        
        Args:
            n_concepts (int): Number of binary concepts at the bottleneck.
            n_classes (int): Number of classes.
            arch: Architecture to use. Allowed values are 'XtoC', 'CtoY', 'XtoCtoY', '
                  'XtoCtoY_sigmoid' or 'XtoY'
            non_linear_ctoy (bool): Whether to use a non-linear CtoY model.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_concepts = n_concepts
        self.arch = arch
        self.use_sigmoid = False
        if arch is enum_utils.Arch.X_TO_C_TO_Y_SIGMOID: 
            self.use_sigmoid = True
            self.arch = enum_utils.Arch.X_TO_C_TO_Y

        # --- X_TO_C part ---
        if self.arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_C_TO_Y, enum_utils.Arch.X_TO_Y]: 
            # load pretrained InceptionV3, remove final layer
            inception = inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
            # inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            num_ftrs = inception.fc.in_features
            inception.fc = nn.Identity() # remove final classification layer 
            inception.AuxLogits = nn.Identity()
            self.base_model = create_feature_extractor(inception, return_nodes={'Mixed_7c': 'features'})

            self.gap = nn.AdaptiveAvgPool2d((1, 1)) # global average pooling

            # Conception prediction layer (or intermediate layer for X_TO_Y)
            activation_fn = nn.ReLU() if self.arch is enum_utils.Arch.X_TO_Y else nn.Identity()
            self.concept_layer = nn.Sequential(nn.Linear(num_ftrs, n_concepts), activation_fn) 

        # --- C_TO_Y part ---
        if self.arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y, enum_utils.Arch.X_TO_Y]: 
            if non_linear_ctoy: 
                self.ctoy_module = nn.Sequential(
                    nn.Linear(n_concepts, 128), 
                    nn.ReLU(), 
                    # nn.Dropout(0.2), # Optional dropout
                    nn.Linear(129, 128), 
                    nn.ReLU(),
                    # nn.Dropout(0.2), # Optional dropout
                    nn.Linear(128, n_classes),
                )
            else: 
                self.ctoy_module = nn.Linear(n_concepts, n_classes)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]: 
        """
        Executes a forward pass through the network. Substitute call()
        
        Args: 
            inputs (torch.Tensor): tensor containing the input batch. 
                    Shapes depends on arch: (N, C, H, W) for image input, (N, n_concepts) for concept input.
                    
        Returns:
            Tuple: containing the model predictions (logits). 
                   Possible return formats:
                   - (concept_logits, )
                   - (class_logits, )
                   - (concept_logits, class_logits)
        """
        if self.arch is enum_utils.Arch.X_TO_C: 
            # input: image 
            # features = self.base_model(inputs)
            features = self.base_model(inputs)['features']
            pooled_features = self.gap(features).flatten(start_dim=1)
            concept_logits = self.concept_layer(pooled_features)
            return (concept_logits, )
        
        elif self.arch is enum_utils.Arch.C_TO_Y:
            # input: concept vector (can be logits or probabilities)
            class_logits = self.ctoy_module(inputs)
            return (class_logits, )
        
        elif self.arch is enum_utils.Arch.X_TO_C_TO_Y: 
            features = self.base_model(inputs)['features']
            pooled_features = self.gap(features).flatten(start_dim=1)
            concept_logits = self.concept_layer(pooled_features) # note: concept layer has no activation here
            
            # input to CtoY can be logits or sigmoid(logits)
            ctoy_input = torch.sigmoid(concept_logits) if self.use_sigmoid else concept_logits
            class_logits = self.ctoy_module(ctoy_input)
            return (concept_logits, class_logits)
        
        elif self.arch is enum_utils.Arch.X_TO_Y: 
            features = self.base_model(inputs)
            pooled_features = self.gap(features).flatten(start_dim=1)
            intermediate_representation = self.concept_layer(pooled_features) # has ReLU activation
            class_logits = self.ctoy_module(intermediate_representation)
            return (class_logits, )
        
        else: 
            raise ValueError(f"Unsupported architecture: {self.arch}")
    
    def get_x_y_from_data(self, data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Unpacks the model inputs and ground truth outputs from data batch.
         
        Args: 
            data: a tuple from DataLoader (image, concept_label, class_label, concept_uncertainty)
        
        Returns:
            x (torch.Tensor): input tensor (image)
            y (Tuple[torch.Tensor, ...]): a tuple of ground truth output tensors.
        """
        image, concept_label, class_label, _ = data # ignore uncertainty for standard training/eval
        
        if self.arch is enum_utils.Arch.X_TO_C: 
            x = image
            y = (concept_label, )
        elif self.arch is enum_utils.Arch.C_TO_Y:
            x = concept_label.float() # esure they are float for linear layer 
            y = (class_label, )
        elif self.arch is enum_utils.Arch.X_TO_C_TO_Y:
            x = image
            y = (concept_label, class_label)
        elif self.arch is enum_utils.Arch.X_TO_Y:
            x = image
            y = (class_label, )
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        return x, y
    

