"""
VascuMap3D - 3D Vascular Network Segmentation and Analysis

A comprehensive pipeline for segmenting and analyzing 3D vascular networks
from microscopy images.

Pipeline Steps:
    1. Image Translation: Brightfield to fluorescence using Pix2Pix 3D GAN
    2. Segmentation: 2D U-Net with 2.5D inference (axial, coronal, sagittal)
    3. Graph Analysis: Skeletonization and morphometric analysis

Modules:
    - models: Neural network architectures (Pix2Pix, U-Net)
    - pipeline: Inference and analysis pipelines

Example:
    >>> from vascumap.pipeline import run, process_directory
    >>> run(input_dir="/path/to/data", model_p2p_path="translation.ckpt",
    ...     model_smp_path="segmentation.pth")
    >>> process_directory("/path/to/data", suffix="_vessel_mask", save_local_metrics=True)
"""

__version__ = "1.1.0"
__author__ = "Luca Rappez"

from . import models
from . import pipeline

__all__ = ['models', 'pipeline']

