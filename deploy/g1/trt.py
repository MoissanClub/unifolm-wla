"""Experimental TensorRT 10 action-velocity head; VLM stays in PyTorch.

Exported inputs have fixed batch, horizon and text length. Padding is masked.
Do not use an engine for motion until per-step and rollout parity are measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def velocity_wrapper(head):
    import torch
    from .inference import use_native_attention
    use_native_attention(head)

    class Velocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, hidden, text_mask, actions, action_mask, timestep):
            h = self.head
            mask = action_mask.unsqueeze(1).expand_as(actions).to(actions.dtype)
            features = h.action_encoder(torch.cat((actions*mask, mask), dim=-1), timestep)
            features = h._add_action_pos_embed(features, actions.device)
            body = torch.ones((actions.shape[0],), device=actions.device, dtype=torch.long)
            features = h._add_embodiment_embed(features, body)
            output = h.model(hidden_states=features, encoder_hidden_states=hidden,
                             encoder_attention_mask=text_mask, timestep=timestep)
            output = h._add_embodiment_embed_dec(output, body)
            return h.action_decoder(output)[:, -h.action_horizon:]

    if head.state_encoder is not None or list(head.embodiment_types) != ["unitree"]:
        raise ValueError("Exporter supports the released state-projector/Unitree head only")
    return Velocity().eval()


class TRTVelocity:
    def __init__(self, engine_path):
        import torch
        import tensorrt as trt
        self.torch, self.trt = torch, trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not load engine; rebuild on this Jetson/TRT version")
        self.context = self.engine.create_execution_context()
        self.inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                       if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        expected = {"hidden", "text_mask", "actions", "action_mask", "timestep"}
        if set(self.inputs) != expected:
            raise ValueError(f"Wrong engine inputs: {self.inputs}")
        self.max_tokens = self.engine.get_tensor_shape("hidden")[1]
        if self.max_tokens < 1:
            raise ValueError("Expected a fixed-shape action head")

    def __call__(self, **values):
        torch, trt = self.torch, self.trt
        dtypes = {trt.float16: torch.float16, trt.float32: torch.float32,
                  trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
        tensors = {}
        for name in self.inputs:
            tensor = values[name].to(dtype=dtypes[self.engine.get_tensor_dtype(name)], device="cuda").contiguous()
            if tuple(tensor.shape) != tuple(self.engine.get_tensor_shape(name)):
                raise ValueError(f"Engine shape mismatch for {name}: {tuple(tensor.shape)}")
            tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
        shape = tuple(self.engine.get_tensor_shape("velocity"))
        output = torch.empty(shape, dtype=dtypes[self.engine.get_tensor_dtype("velocity")], device="cuda")
        self.context.set_tensor_address("velocity", output.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        # Synchronize before local cast buffers can be freed/reused on another stream.
        stream.synchronize()
        return output

    def predict(self, hidden, text_mask, action_mask, config):
        torch = self.torch
        if hidden.shape[1] > self.max_tokens:
            raise ValueError(f"Prompt has {hidden.shape[1]} tokens; engine supports {self.max_tokens}")
        padding = self.max_tokens-hidden.shape[1]
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, padding))
        text_mask = torch.nn.functional.pad(text_mask, (0, padding), value=False)
        actions = torch.randn((1, int(config.action_horizon), int(config.action_dim)), device=hidden.device)
        mask = action_mask.float().unsqueeze(1)
        actions *= mask
        steps = int(config.num_inference_timesteps)
        for step in range(steps):
            timestep = torch.tensor([int(step/steps*int(config.num_timestep_buckets))], device=hidden.device)
            velocity = self(hidden=hidden, text_mask=text_mask, actions=actions,
                            action_mask=action_mask, timestep=timestep)
            actions = (actions + velocity.float()/steps)*mask
        return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", default=1024, type=int)
    args = parser.parse_args()
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from unifolm_wla.model.framework.share_tools import dict_to_namespace, read_mode_config
    from unifolm_wla.model.modules.action_model.MMDiT_ActionHeader import get_action_model

    config, _ = read_mode_config(args.checkpoint)
    cfg = dict_to_namespace(config)
    with init_empty_weights():
        head = get_action_model(cfg)
    with safe_open(args.checkpoint, framework="pt", device="cpu") as weights:
        keys = {k.removeprefix("action_model.") for k in weights.keys() if k.startswith("action_model.")}
        if keys != set(head.state_dict()):
            raise ValueError("Action-head architecture mismatch")
        for key in keys:
            set_module_tensor_to_device(head, key, "cpu", value=weights.get_tensor("action_model."+key),
                                        dtype=torch.float32)
    head.eval().requires_grad_(False)
    wrapper = velocity_wrapper(head)
    example = (
        torch.zeros(1, args.tokens, int(cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim)),
        torch.ones(1, args.tokens, dtype=torch.bool),
        torch.zeros(1, head.action_horizon, head.action_output_dim),
        torch.ones(1, head.action_output_dim), torch.zeros(1, dtype=torch.int64),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, example, str(args.out),
                      input_names=["hidden", "text_mask", "actions", "action_mask", "timestep"],
                      output_names=["velocity"], opset_version=18, dynamo=True, external_data=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "status": "exported, parity not yet established", "tokens": args.tokens,
        "checkpoint": str(Path(args.checkpoint).resolve()), "torch": torch.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()
