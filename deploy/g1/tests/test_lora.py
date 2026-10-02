import pytest


def test_merge_preserves_adapter_and_nonadapter_updates():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from deploy.g1.finalize import merge_lora_layers
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    peft.inject_adapter_in_model(peft.LoraConfig(r=2, lora_alpha=4, target_modules=["0"]), model)
    with torch.no_grad():
        model[0].lora_B["default"].weight.fill_(.2)
        model[1].weight.add_(.7)  # a trained non-LoRA component must survive
    model.eval()
    x = torch.randn(5, 3)
    before = model(x).detach()
    merge_lora_layers(model)
    torch.testing.assert_close(model(x), before)
    assert not any("lora_" in name for name in model.state_dict())
