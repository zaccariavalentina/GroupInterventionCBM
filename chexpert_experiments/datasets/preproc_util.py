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

"""Common preprocessing utilites for datasets (PyTorch version)."""

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import random


def center_crop_and_resize(image, height, width, crop_proportion): 
    """Crops to center of image and rescales to desired size.

    Args:
        image: Input PIL Image or Tensor (C, H, W).
        height: Target height.
        width: Target width.
        crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
        A (height, width) Tensor holding a central crop of `image`.
    """
    if isinstance(image, torch.Tensor): 
        if image.ndim == 3: 
            image_height, image_width = image.shape[1], image.shape[2]
        elif image.ndim == 2: 
            image_height, image_width = image.shape[0], image.shape[1]
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")
    else:
        # PIL Image
        image_height, image_width = image.size
    
    aspect_ratio = width / height
    image_width_float = float(image_width)
    image_height_float = float(image_height)

    if aspect_ratio > image_width_float / image_height_float:
        # wider than image 
        crop_height = int(round(crop_proportion / aspect_ratio * image_width_float))
        crop_width = int(round(crop_proportion * aspect_ratio * image_height_float))
    else: 
        # image wider than aspect ration
        crop_height = int(round(crop_proportion * image_height_float))
        crop_width = int(round(crop_proportion * aspect_ratio * image_height_float))
    
    # Ensure crop dimensions are valid
    crop_height = min(image_height, crop_height)
    crop_width = min(image_width, crop_width)

    # F.center_crop expects output size, we calculated crop size, so we use crop directly
    # Calculate top-left corner for cropping
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2

    # Crop 
    image = F.crop(image, offset_height, offset_width, crop_height, crop_width)

    # Resize
    image = F.resize(image, [height, width], InterpolationMode.BICUBIC)

    return image 


def random_brightness(image, max_delta):
    """Applies random brightness adjustment."""
    if max_delta < 0:
        raise ValueError("max_delta must be non-negative.")
    if max_delta == 0:
        return image

    factor = random.uniform(max(0.0, 1.0 - max_delta), 1.0 + max_delta)
    return F.adjust_brightness(image, factor)

def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0):
    """Distorts the color of the image (jittering order is random).

    Args:
        image: The input image tensor (C, H, W) with values in [0, 1].
        brightness: A float, specifying the max delta for brightness adjustment.
        contrast: A float, specifying the max delta for contrast adjustment.
        saturation: A float, specifying the max delta for saturation adjustment.
        hue: A float, specifying the max delta for hue adjustment [-0.5, 0.5].
    Returns:
        The distorted image tensor.
    """
    transforms = []
    if brightness > 0:
        transforms.append(lambda img: random_brightness(img, brightness))
    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        transforms.append(lambda img: F.adjust_contrast(img, contrast_factor))
    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        transforms.append(lambda img: F.adjust_saturation(img, saturation_factor))
    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
        transforms.append(lambda img: F.adjust_hue(img, hue_factor))

    random.shuffle(transforms)

    jittered_image = image
    for transform in transforms:
        jittered_image = transform(jittered_image)

    # Clip to ensure values remain in [0, 1] range
    jittered_image = torch.clamp(jittered_image, 0., 1.)
    return jittered_image