# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()

def is_xpu_available() -> bool:
    """Check Intel XPU availability"""
    try:
        if hasattr(torch, "xpu") and callable(getattr(torch.xpu, "is_available", None)):
            return torch.xpu.is_available()
        return False
    except ImportError:
        return False


is_xpu_available = is_xpu_available()


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    if is_cuda_available:
        return "CUDA_VISIBLE_DEVICES"
    elif is_xpu_available:
        return "ZE_AFFINITY_MASK"
    else:
        return "ASCEND_RT_VISIBLE_DEVICES"


def get_attention_implementation() -> str:
    """Return the appropriate attention implementation based on device type."""
    if is_cuda_available:
        return "flash_attention_2"  # Use Flash Attention 2 for CUDA
    elif is_xpu_available:
        return "sdpa"  # Use PyTorch SDPA for Intel XPU (Flash Attention 2 not available)
    elif is_npu_available:
        return "sdpa"  # Use PyTorch SDPA for NPU
    else:
        return "eager"  # Use eager implementation for CPU


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently supports CPU, CUDA, NPU, and XPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_xpu_available:
        device = "xpu"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.xpu if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    if is_cuda_available:
        return "nccl"
    elif is_xpu_available:
        return "xccl"  # XPU uses XCCL backend
    elif is_npu_available:
        return "hccl"
    else:
        raise RuntimeError(f"No available ccl backend found on device type {get_device_name()}.")


def empty_cache() -> None:
    """Device-agnostic cache clearing.
    
    Clears the memory cache for the current device type (CUDA, XPU, NPU).
    This is a wrapper that calls the appropriate cache clearing function
    based on the available device.
    """
    if is_cuda_available:
        torch.cuda.empty_cache()
    elif is_xpu_available:
        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()
        else:
            logger.debug("torch.xpu.empty_cache() not available")
    elif is_npu_available:
        if hasattr(torch.npu, 'empty_cache'):
            torch.npu.empty_cache()
        else:
            logger.debug("torch.npu.empty_cache() not available")
    else:
        # CPU doesn't have a cache to clear
        pass


def set_expandable_segments(enable: bool) -> None:
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
