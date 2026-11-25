"""Masking of torch tensors, through the array API on the tensor's device."""

import torch

from .base import impute, impute_array, select, select_array

impute.register(torch.Tensor, impute_array)
select.register(torch.Tensor, select_array)
