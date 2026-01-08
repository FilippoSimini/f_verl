import os
import sys
import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import contextlib
import warnings
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_attention_implementation,
)
from verl.utils.fsdp_utils import (
    init_fn,
    get_init_weight_context_manager,
    get_fsdp_wrap_policy,
    create_device_mesh,
    get_sharding_strategy,
    CPUOffload,
)
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.model import (
    get_generation_config,
    update_model_config,
    print_model_size,
    log_gpu_memory_usage,
)
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import FSDPEngineConfig, HFModelConfig, ActorConfig # Import ActorConfig

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

def _build_model_optimizer_reproducer(
    model_path,
    fsdp_config: FSDPEngineConfig,
    optim_config,
    override_model_config,
    use_remove_padding=False,
    use_fused_kernels=False,
    enable_gradient_checkpointing=False,
    trust_remote_code=False,
    use_liger=False,
    role="actor",
    enable_activation_offload=False,
    device_mesh=None,
    ulysses_sequence_parallel_size=1,
    is_lora=False,
    rank=0,
    world_size=1,
    config_obj=None, # Pass the full config_obj here
):
    assert role in ["actor", "ref"]

    log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=print)
    local_path = model_path

    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

    # Dummy generation_config, not used in this reproducer's forward pass
    generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

    torch_dtype = fsdp_config.get("model_dtype", None)
    if torch_dtype is None:
        torch_dtype = torch.float32 # Defaulting to float32 if not specified
    else:
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

    actor_model_config = AutoConfig.from_pretrained(
        local_path, trust_remote_code=trust_remote_code, attn_implementation=get_attention_implementation()
    )

    if ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
        actor_model_config.vision_config._attn_implementation = "eager"

    if getattr(actor_model_config, "model_type", None) == "kimi_vl":
        actor_model_config.text_config.topk_method = "greedy"

    override_config_kwargs = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    override_config_kwargs.update(override_model_config)
    update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
    if rank == 0:
        print(f"Model config after override: {actor_model_config}")

    init_context = get_init_weight_context_manager(
        use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=device_mesh
    )

    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        has_remote_code = hasattr(actor_model_config, "auto_map") and any(
            actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
        )
        if has_remote_code:
            auto_class = next(
                k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
            )
            match auto_class:
                case "AutoModelForVision2Seq":
                    actor_module_class = AutoModelForVision2Seq
                case "AutoModelForCausalLM":
                    actor_module_class = AutoModelForCausalLM
                case "AutoModelForImageTextToText":
                    actor_module_class = AutoModelForImageTextToText
                case _:
                    actor_module_class = AutoModel
        else:
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                actor_module_class = AutoModelForCausalLM
            elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                actor_module_class = AutoModelForImageTextToText
            else:
                actor_module_class = AutoModel

        actor_module = actor_module_class.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=actor_model_config,
            trust_remote_code=trust_remote_code,
        )

        if use_liger:
            # Dummy import for liger_kernel if not available
            class _LigerKernelDummy:
                def _apply_liger_kernel_to_instance(self, model):
                    print("Liger kernel not available, skipping application.")
            _apply_liger_kernel_to_instance = _LigerKernelDummy()._apply_liger_kernel_to_instance
            _apply_liger_kernel_to_instance(model=actor_module)

        fused_kernel_options = config_obj.model.get("fused_kernel_options", None)
        fused_kernels_backend = (
            fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
        )

        apply_monkey_patch(
            model=actor_module,
            use_remove_padding=use_remove_padding,
            ulysses_sp_size=ulysses_sequence_parallel_size,
            use_fused_kernels=use_fused_kernels,
            fused_kernels_backend=fused_kernels_backend,
        )

        actor_module.to(torch_dtype)

        if enable_gradient_checkpointing:
            actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if is_lora:
        print("Applying LoRA to actor module")
        # Simplified LoRA application for reproducer
        # In a real scenario, this would involve peft.get_peft_model
        pass

    dist.barrier()

    if rank == 0:
        print_model_size(actor_module)

    log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=print)

    mixed_precision_config = fsdp_config.get("mixed_precision", None)
    if mixed_precision_config is not None:
        param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
        reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
        buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
    else:
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32

    mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

    auto_wrap_policy = get_fsdp_wrap_policy(
        module=actor_module,
        config=fsdp_config.get("wrap_policy", None),
        is_lora=is_lora,
    )

    if rank == 0:
        print(f"wrap_policy: {auto_wrap_policy}")

    fsdp_mesh = device_mesh
    sharding_strategy = get_sharding_strategy(fsdp_mesh)

    cpu_offload = None # For actor, cpu_offload is None in fsdp_workers.py

    actor_module_fsdp = FSDP(
        actor_module,
        cpu_offload=cpu_offload,
        param_init_fn=init_fn,
        auto_wrap_policy=auto_wrap_policy,
        device_id=get_device_id(),
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_mesh=device_mesh,
        use_orig_params=fsdp_config.get("use_orig_params", False),
        forward_prefetch=fsdp_config.get("forward_prefetch", False),
    )

    # Dummy optimizer and lr_scheduler for reproducer
    actor_optimizer = torch.optim.AdamW(actor_module_fsdp.parameters(), lr=optim_config.lr)
    actor_lr_scheduler = None

    return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config


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

    # Create dummy config object
    config_obj = OmegaConf.create({
        "actor": {
            "fsdp_config": {
                "fsdp_size": -1, # Assuming full shard for single GPU
                "use_orig_params": False,
                "forward_prefetch": False,
                "param_offload": False,
                "optimizer_offload": False,
                "model_dtype": "bf16", # Matching the current training script
                "wrap_policy": {"min_num_params": 0}, # Default
            },
            "optim": {
                "lr": 1e-6,
                "total_training_steps": 3, # Dummy value
                "lr_warmup_steps": -1,
                "lr_warmup_steps_ratio": 0.0,
                "lr_scheduler_type": "constant",
                "min_lr_ratio": 0.0,
                "num_cycles": 0.5,
                "optimizer": "AdamW",
                "optimizer_impl": "torch.optim",
                "override_optimizer_config": None,
                "clip_grad": 1.0,
            },
        },
        "model": {
            "path": model_name,
            "trust_remote_code": False,
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "enable_gradient_checkpointing": False,
            "use_liger": False,
            "enable_activation_offload": False,
            "override_config": {},
            "lora_adapter_path": None,
            "lora_rank": 0,
            "lora_alpha": 16,
            "target_modules": "all-linear",
            "exclude_modules": None,
            "use_shm": False,
            "fused_kernel_options": {"impl_backend": "torch"}, # Default
        },
        "nccl_timeout": 600, # Default from fsdp_workers.py
    })

    # Build device mesh
    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=config_obj.actor.fsdp_config.fsdp_size)

    # Call the reproducer's build model optimizer
    actor_module_fsdp, actor_optimizer, _, actor_model_config = _build_model_optimizer_reproducer(
        model_path=model_name,
        fsdp_config=omega_conf_to_dataclass(config_obj.actor.fsdp_config, dataclass_type=FSDPEngineConfig),
        optim_config=config_obj.actor.optim,
        override_model_config=config_obj.model.override_config,
        use_remove_padding=config_obj.model.use_remove_padding,
        use_fused_kernels=config_obj.model.use_fused_kernels,
        enable_gradient_checkpointing=config_obj.model.enable_gradient_checkpointing,
        trust_remote_code=config_obj.model.trust_remote_code,
        use_liger=config_obj.model.use_liger,
        role="actor",
        enable_activation_offload=config_obj.model.enable_activation_offload,
        device_mesh=device_mesh,
        ulysses_sequence_parallel_size=1, # Assuming 1 for reproducer
        is_lora=False, # Assuming no LoRA for reproducer
        rank=rank,
        world_size=world_size,
        config_obj=config_obj,
    )

    # Load state dict into base model
    print(f"[{rank}] Loading model state from {model_file}")
    state = torch.load(model_file, map_location="cpu")
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    actor_module_fsdp.load_state_dict(new_state, strict=False)

    # Prepare optimizer and load its state
    print(f"[{rank}] Initializing optimizer and loading state from {optim_file}")
    opt_state = torch.load(optim_file, map_location="cpu", weights_only=False)
    actor_optimizer.load_state_dict(opt_state)

    actor_module_fsdp.train()

    # Move inputs to device
    print(f"[{rank}] Preparing inputs on device")
    input_ids = input_dict["input_ids"].to(device)
    attention_mask = input_dict["attention_mask"].to(device)
    position_ids = input_dict["position_ids"].to(device)
    
    # Forward pass
    torch.set_printoptions(threshold=1000, edgeitems=2) 
    print(f"\n~~~~~~~ input_ids finite? {torch.isfinite(input_ids).all()}\n\t{input_ids}\n")
    print(f"\n~~~~~~~ position_ids finite? {torch.isfinite(position_ids).all()}\n\t{position_ids}\n")
    print(f"[{rank}] Running forward pass")
    
    # Replicate autocast context from dp_actor.py
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    
    with autocast_ctx:
        outputs = actor_module_fsdp(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    
    logits = outputs.logits
    if not torch.isfinite(logits).all():
        print(f"[{rank}] ERROR: NaNs detected in logits after forward pass!")
        sys.exit(1)
    
    # The rest of the loss calculation is not strictly necessary for NaN reproduction in forward pass
    # but kept for completeness if needed later.
    logits = logits[..., :-1, :].contiguous()
    shift_logits = logits.view(-1, actor_model_config.vocab_size)
    labels = input_ids[:, 1:].contiguous().view(-1) # This was input_ids[:, 1:] before, now it should be responses

    loss = torch.nn.functional.cross_entropy(
        shift_logits, labels, reduction="none"
    )
    # Simplified loss_mask handling for reproducer
    denom = loss.numel()
    loss = loss.sum() / denom
    print(f"[{rank}] Computed loss: {loss.item()}")

    # Finalize
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
