import time
import sys
import torch


def _log_nonfinite(name, obj, logger_fn=print, max_indices=5):
    """
    Log finiteness diagnostics for a tensor or a nested structure containing tensors.

    - Prints a single-line summary with the prefix "FINITE_CHECK" for each tensor.
    - Returns True if any non-finite values are detected in the given object; otherwise False.

    Args:
        name: Label for the object being checked (used in log prefix).
        obj: A torch.Tensor, dict of tensors, or list/tuple of tensors to check.
        logger_fn: Callable used for logging output (default: print).
        max_indices: Max number of non-finite indices to display for context.
    """

    def _log_one(nm, t: torch.Tensor) -> bool:
        # Best-effort conversion for DTensor-like wrappers
        if hasattr(t, "full_tensor"):
            try:
                t = t.full_tensor()
            except Exception:
                pass
        # Detach and move to CPU for stable diagnostics
        try:
            cpu_t = t.detach().to("cpu")
        except Exception:
            cpu_t = t.detach()

        total = cpu_t.numel()
        if total == 0:
            logger_fn(
                f"FINITE_CHECK [{nm}]: empty tensor; shape={tuple(cpu_t.shape)}; dtype={cpu_t.dtype}; device={t.device}"
            )
            return False

        # Determine whether dtype supports non-finite values
        is_float_or_complex = cpu_t.is_floating_point() or cpu_t.dtype in (torch.complex64, torch.complex128)
        if is_float_or_complex:
            finite_mask = torch.isfinite(cpu_t)
            nonfinite_mask = ~finite_mask
            nan_count = int(torch.isnan(cpu_t).sum().item())
            inf_mask = torch.isinf(cpu_t)
            inf_count = int(inf_mask.sum().item())
        else:
            # Integers and booleans have no NaN/Inf; treat all values as finite
            finite_mask = torch.ones_like(cpu_t, dtype=torch.bool)
            nonfinite_mask = torch.zeros_like(cpu_t, dtype=torch.bool)
            nan_count = 0
            inf_mask = torch.zeros_like(cpu_t, dtype=torch.bool)
            inf_count = 0

        num_nonfinite = int(nonfinite_mask.sum().item())
        num_finite = int(total - num_nonfinite)
        pos_inf = int((cpu_t[inf_mask] > 0).sum().item()) if inf_count > 0 else 0
        neg_inf = inf_count - pos_inf

        # Finite stats
        if num_finite > 0:
            finite_vals = cpu_t[finite_mask]
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())
            mean_val = float(finite_vals.float().mean().item())
        else:
            min_val = float("nan")
            max_val = float("nan")
            mean_val = float("nan")

        shape = tuple(cpu_t.shape)
        dtype = str(cpu_t.dtype)
        device = str(t.device)

        if num_nonfinite == 0:
            logger_fn(
                f"FINITE_CHECK [{nm}]: all finite; count={total}; stats(min={min_val:.6g}, max={max_val:.6g}, mean={mean_val:.6g}); "
                f"shape={shape}; dtype={dtype}; device={device}"
            )
            return False
        else:
            idxs = nonfinite_mask.nonzero(as_tuple=False)
            sample = idxs[:max_indices].tolist()
            logger_fn(
                f"FINITE_CHECK [{nm}]: NON-FINITE DETECTED; total={total}; nonfinite={num_nonfinite}; nan={nan_count}; "
                f"inf={inf_count} (+inf={pos_inf}, -inf={neg_inf}); finite_count={num_finite}; "
                f"finite_stats(min={min_val:.6g}, max={max_val:.6g}, mean={mean_val:.6g}); first_nonfinite_indices={sample}; "
                f"shape={shape}; dtype={dtype}; device={device}"
            )
            return True

    detected = False
    if isinstance(obj, torch.Tensor):
        detected = _log_one(name, obj) or detected
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                detected = _log_one(f"{name}.{k}", v) or detected
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, torch.Tensor):
                detected = _log_one(f"{name}[{i}]", v) or detected
    else:
        logger_fn(f"FINITE_CHECK [{name}]: unsupported type {type(obj).__name__}")
    return detected


def save_debug_states(batch, fsdp_model, optimizer, prefix="bad", exit_after_save=True):
    """
    Check batch inputs and model parameters for non-finite values,
    and if both are finite, save the inputs, model state, and optimizer state.

    Args:
        batch: TensorDict or dict of input tensors.
        fsdp_model: the FSDP-wrapped model instance.
        optimizer: optimizer used for training.
        prefix: prefix string for saved filenames.
        exit_after_save: whether to exit the process after saving (default: True).
    """
    # Capture timestamp for filenames
    timestamp = int(time.time())

    # Convert batch to simple dict of CPU tensors
    input_dict = {k: v.cpu() for k, v in batch.items() if hasattr(v, 'cpu')}

    # Check inputs for any non-finite values
    input_has_nonfinite = any(
        not torch.isfinite(v).all() for v in input_dict.values() if isinstance(v, torch.Tensor)
    )

    # Check model parameters for any non-finite values
    model_has_nonfinite = any(
        not torch.isfinite(p).all() for p in fsdp_model.parameters()
    )

    saved = False
    if not input_has_nonfinite and not model_has_nonfinite:
        # Save tensors for minimal reproducer
        torch.save(input_dict, f"{prefix}_input_{timestamp}.pt")
        torch.save(fsdp_model.state_dict(), f"{prefix}_model_{timestamp}.pt")
        torch.save(optimizer.state_dict(), f"{prefix}_optim_{timestamp}.pt")
        saved = True
    else:
        print(
            f"WARN: Skipping save because input_has_nonfinite={input_has_nonfinite}, "
            f"model_has_nonfinite={model_has_nonfinite}"
        )


    # Optionally exit after saving
    if saved and exit_after_save:
        print(f"Saved debug states with prefix '{prefix}' and exiting.")
        sys.exit(1)
