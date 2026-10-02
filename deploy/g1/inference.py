"""Memory-conscious BF16 inference; no robot SDK imports or robot commands."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
from .contract import IMAGE_SIZE


def use_native_attention(model):
    from diffusers.models.attention_dispatch import AttentionBackendName
    for module in model.modules():
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_attention_backend"):
            processor._attention_backend = AttentionBackendName.NATIVE


def load_model(checkpoint, device="cuda:0", skip_action=False):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from unifolm_wla.model.framework.base_framework import build_framework
    from unifolm_wla.model.framework.share_tools import dict_to_namespace, read_mode_config

    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix != ".safetensors":
        raise ValueError("Use the released full .safetensors checkpoint")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A working CUDA GPU with BF16 support is required for model inference")
    config, statistics = read_mode_config(checkpoint)
    tokenizer = checkpoint.parents[1] / "tokenizer"
    if not (tokenizer / "config.json").exists():
        raise FileNotFoundError(f"Missing local tokenizer bundle: {tokenizer}")
    cfg = dict_to_namespace(config)
    cfg.framework.qwenvl.base_vlm = str(tokenizer)
    cfg.framework.qwenvl.attn_implementation = "sdpa"
    cfg.framework.robot_state_projector.load_from = str(tokenizer)
    cfg.trainer.pretrained_checkpoint = None
    # Build parameters on meta, then stream one weight at a time to the device.
    # Avoid allocating a random full model plus a full state_dict in Jetson RAM.
    with init_empty_weights():
        model = build_framework(cfg)
    use_native_attention(model)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
        expected, supplied = set(model.state_dict()), set(weights.keys())
        if expected != supplied:
            raise ValueError(f"Checkpoint architecture mismatch: missing={sorted(expected-supplied)[:10]}, "
                             f"unexpected={sorted(supplied-expected)[:10]}")
        for key in weights.keys():
            if skip_action and key.startswith("action_model."):
                continue
            tensor = weights.get_tensor(key)
            dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
            set_module_tensor_to_device(model, key, device, value=tensor, dtype=dtype)
            del tensor
    if skip_action:
        del model.action_model
    # Nonpersistent buffers were constructed on CPU; parameters are already on GPU.
    model.to(device=device)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Unloaded meta parameter")
    model.norm_stats = statistics
    model.eval().requires_grad_(False)
    gc.collect()
    return model


class Policy:
    def __init__(self, checkpoint, image_size=IMAGE_SIZE, engine=None):
        import torch
        self.torch = torch
        self.model = load_model(checkpoint, skip_action=bool(engine))
        self.image_size = image_size
        self.engine = None
        if engine:
            from .trt import TRTVelocity
            self.engine = TRTVelocity(engine)
            self.head_config = self.model.config.framework.action_model
            self.horizon = self.model.action_horizon
            # VLM remains PyTorch; only the action velocity network uses TensorRT.
            gc.collect()
            torch.cuda.empty_cache()

    def encode(self, example):
        from PIL import Image
        torch, model = self.torch, self.model
        images = [Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in example["image"]]
        if self.image_size:
            h, w = self.image_size
            images = [im.resize((w, h), Image.Resampling.BILINEAR) for im in images]
        inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=[images], instructions=[example["lang"]],
            image_roles=[example["image_roles"]], arm_types=[example["arm_type"]],
            add_generation_prompt=False,
        )
        states = model._build_projector_state([example])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.qwen_vl_interface(
                **inputs, robot_states=states, output_hidden_states=True,
                return_dict=True, use_cache=False, logits_to_keep=1,
            )
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].bool()
        del outputs, inputs
        return hidden, mask

    def predict(self, example, seed=None):
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            hidden, text_mask = self.encode(example)
            torch.cuda.synchronize()
            encoded = time.perf_counter()
            mask = torch.as_tensor(example["action_mask"], device=hidden.device).unsqueeze(0)
            if self.engine:
                result = self.engine.predict(hidden, text_mask, mask, self.head_config)
            else:
                body = self.model._build_body_type_ids([example], hidden.device)
                # Upstream's FP32 autocast disables autocast; a BF16 model then
                # receives FP32 masked actions. Keep the actual head under BF16 autocast.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = self.model.action_model.predict_action(
                        hidden, None, action_mask=mask, encoder_attention_mask=text_mask, body_type_ids=body,
                    )
            pred = result[0].float().cpu().numpy()
        torch.cuda.synchronize()
        end = time.perf_counter()
        metrics = {
            "encode_ms": (encoded-start)*1000, "head_ms": (end-encoded)*1000,
            "total_ms": (end-start)*1000, "tokens": int(hidden.shape[1]),
            "allocated_gib": torch.cuda.memory_allocated()/2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        }
        return pred, metrics
