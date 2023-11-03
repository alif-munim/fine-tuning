"""
A minimal implementaion of LoRA.
"""

import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class LoRAParametrization(nn.Module):
    
    def __init__(self):
        super()__init__()
        
    def _dropout(self, A):
        
    def lora_forward(self, X):
        
    def forward(self, X):
        
    def disable_lora(self):
        
    @classmethod
    def from_linear():
        
    @classmethod:
    def from_conv2d():
    
    @classmethod():
    def from_embedding:
        
        
def apply_lora(layer):
    """Add LoRA parametrization to a layer (used with model.apply)."""
    
def add_lora(model):
    """Add LoRA parametrization to all layers in a model."""
    
def add_lora_by_name(model):
    """Add LoRA parametrization to specific layers in a model by name."""
    
def merge_lora(model):
    """Merge LoRA parametrization to all layers in a model. Removes parametrization."""
    
def remove_lora(model):
    """Remove LoRA paramterization from all layers in a model."""