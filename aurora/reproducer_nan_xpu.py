import os
import sys
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

def init_dist():
    # Initialize torch distributed using CCL or NCCL/XCCL depending on device
    if not dist.is_initialized():
        rank = int(os.environ.get("CCL_LOCAL_RANK", os.environ.get("RANK", "0")))
        world_size = int(os.environ.get("CCL_LOCAL_SIZE", os.environ.get("WORLD_SIZE", "1")))
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        if torch.xpu.is_available():
            backend = "xccl"
        elif torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.xpu.is_available():
            torch.xpu.set_device(rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(rank)
    return dist.get_rank(), dist.get_world_size()


def main():
    if len(sys.argv) != 5:
        print("Usage: python reproducer.py bad_input.pt bad_model.pt bad_optim.pt model_name_or_path")
        sys.exit(1)
    input_file, model_file, optim_file, model_name = sys.argv[1:]

    rank, world_size = init_dist()
    # Determine device string
    if torch.xpu.is_available():
        device_type = "xpu"
    elif torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"
    if device_type == "xpu":
        dev_idx = torch.xpu.current_device()
    elif device_type == "cuda":
        dev_idx = torch.cuda.current_device()
    else:
        dev_idx = 0
    device = f"{device_type}:{dev_idx}"
    print(f"[{rank}/{world_size}] Using device: {device}")

    # Load offending input batch
    print(f"[{rank}] Loading input from {input_file}")
    input_dict = torch.load(input_file, map_location="cpu")

    # Load base model configuration and unwrapped model
    print(f"[{rank}] Building base model from {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Load state dict into base model
    print(f"[{rank}] Loading model state from {model_file}")
    state = torch.load(model_file, map_location="cpu")
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    base_model.load_state_dict(new_state, strict=False)

    # Wrap model with FSDP
    print(f"[{rank}] Wrapping model with FSDP")
    # Choose dtype for parameters
    model_dtype = torch.bfloat16 #if device_type == "xpu" else torch.float32
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    mixed_precision = MixedPrecision(
        param_dtype=model_dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    model = FSDP(
        base_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device,
        auto_wrap_policy=auto_wrap_policy,
        sync_module_states=True,
    )
    model.train()

    # Prepare optimizer and load its state
    print(f"[{rank}] Initializing optimizer and loading state from {optim_file}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    opt_state = torch.load(optim_file, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(opt_state)

    # Move inputs to device
    print(f"[{rank}] Preparing inputs on device")
    input_ids = input_dict["input_ids"].to(device)
    attention_mask = input_dict["attention_mask"].to(device)
    position_ids = input_dict["position_ids"].to(device)
    loss_mask = input_dict.get("loss_mask")
    if loss_mask is not None:
        loss_mask = loss_mask[:, 1:].reshape(-1).to(device)

    # Forward pass
    torch.set_printoptions(threshold=1000, edgeitems=2) 
    print(f"\n~~~~~~~ input_ids finite? {torch.isfinite(input_ids).all()}\n\t{input_ids}\n")
    print(f"\n~~~~~~~ position_ids finite? {torch.isfinite(position_ids).all()}\n\t{position_ids}\n")
    print(f"[{rank}] Running forward pass")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    print(f"\n~~~~~~~ logits finite? {torch.isfinite(outputs.logits).all()}\n")

    logits = outputs.logits[..., :-1, :].contiguous()
    shift_logits = logits.view(-1, model.config.vocab_size)
    labels = input_ids[:, 1:].contiguous().view(-1)

    loss = torch.nn.functional.cross_entropy(
        shift_logits, labels, reduction="none"
    )
    if loss_mask is not None:
        loss = loss * loss_mask
        denom = loss_mask.sum().clamp_min(1)
    else:
        denom = loss.numel()
    loss = loss.sum() / denom
    print(f"[{rank}] Computed loss: {loss.item()}")

    ## Backward and gradient norm
    #print(f"[{rank}] Running backward")
    #optimizer.zero_grad()
    #loss.backward()
    #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #print(f"[{rank}] grad_norm: {grad_norm}")

    # Finalize
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
