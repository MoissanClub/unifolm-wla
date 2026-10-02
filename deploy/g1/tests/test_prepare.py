from deploy.g1.prepare_data import prepare


def test_config_reuses_wbt_schema(tmp_path):
    config = prepare(tmp_path, tmp_path / "cache")
    source = config["datasets"][0]
    assert source["name"] == "UnifoLM_WBT"
    assert source["state_keys"]["base_rot"] == "observation.state.state_base_rot"
    assert len(source["image_keys"]) == 3
    assert "base_pose" in source["action_keys"]


def test_stationary_head_only_ablation(tmp_path):
    config = prepare(tmp_path, tmp_path / "cache", stationary_arms=True, head_only=True)
    source = config["datasets"][0]
    assert len(source["image_keys"]) == 1
    assert set(source["action_keys"]) == {"left_ee_pose", "right_ee_pose", "left_fig6d", "right_fig6d"}
