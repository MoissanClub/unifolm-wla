import pytest


@pytest.fixture
def small_head(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from omegaconf import OmegaConf
    from unifolm_wla.model.modules.action_model import MMDiT_ActionHeader as mod
    monkeypatch.setitem(mod.DiTConfig, "test", {
        "input_embedding_dim": 32, "attention_head_dim": 8, "num_attention_heads": 4,
    })
    cfg = OmegaConf.create({"framework": {"action_model": {
        "action_model_type": "test", "hidden_size": 32, "state_dim": 0, "action_dim": 54,
        "action_horizon": 3, "add_pos_embed": True, "max_seq_len": 64,
        "num_inference_timesteps": 4, "num_timestep_buckets": 1000,
        "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": .999,
        "embodiment_types": ["unitree"], "diffusion_model_cfg": {
            "num_layers": 1, "cross_attention_dim": 16, "output_dim": 32,
            "dropout": 0.0, "final_dropout": False, "interleave_self_attention": True,
            "norm_type": "ada_norm", "positional_embeddings": None,
        },
    }}})
    return mod.get_action_model(cfg).eval()


def test_split_velocity_preserves_flow_rollout(small_head):
    import torch
    from deploy.g1.trt import velocity_wrapper
    head = small_head
    hidden, text_mask = torch.randn(1, 5, 16), torch.tensor([[True, True, True, False, False]])
    mask = torch.ones(1, 54)
    mask[:, 6] = 0
    body = torch.ones(1, dtype=torch.long)
    torch.manual_seed(8)
    reference = head.predict_action(hidden, None, mask, text_mask, body)
    torch.manual_seed(8)
    actions = torch.randn(1, 3, 54)*mask[:, None]
    wrapper = velocity_wrapper(head)
    for i in range(4):
        velocity = wrapper(hidden, text_mask, actions, mask, torch.tensor([i*250]))
        actions = (actions+velocity/4)*mask[:, None]
    torch.testing.assert_close(actions, reference)


def test_velocity_onnx_export_and_reference_parity(small_head, tmp_path):
    import torch
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from onnx.reference import ReferenceEvaluator
    from deploy.g1.trt import velocity_wrapper
    wrapper = velocity_wrapper(small_head).requires_grad_(False)
    inputs = (torch.randn(1, 5, 16), torch.tensor([[True, True, True, False, False]]),
              torch.randn(1, 3, 54), torch.ones(1, 54), torch.tensor([250]))
    names = ["hidden", "text_mask", "actions", "action_mask", "timestep"]
    path = tmp_path / "velocity.onnx"
    torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=["velocity"],
                      opset_version=18, dynamo=True, external_data=True)
    model = onnx.load(str(path))
    onnx.checker.check_model(model, full_check=True)
    prediction = ReferenceEvaluator(model).run(None, {k: v.numpy() for k, v in zip(names, inputs)})[0]
    torch.testing.assert_close(torch.from_numpy(prediction), wrapper(*inputs), rtol=1e-4, atol=1e-5)
