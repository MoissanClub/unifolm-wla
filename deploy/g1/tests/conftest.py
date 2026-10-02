import json
from pathlib import Path

import numpy as np
import pytest

# This workstation's conda Pinocchio requires loading before torch (libstdc++).
try:
    import pinocchio  # noqa: F401
except ImportError:
    pass

from deploy.g1.contract import PROTOCOL


@pytest.fixture
def observation():
    return {
        "protocol": PROTOCOL, "frame_id": "frame-1", "instruction": "pick up the can",
        "captured_monotonic": 100.0, "motors": np.zeros(35),
        "images": {role: np.zeros((48, 64, 3), np.uint8) for role in
                   ("head_left", "cam_wrist_left", "cam_wrist_right")},
        "state": {
            "left_ee_pose": np.array([0.3, 0.25, 0.2, 0, 0, 0]),
            "right_ee_pose": np.array([0.3, -0.25, 0.2, 0, 0, 0]),
            "left_fig6d": np.full(6, 0.5), "right_fig6d": np.full(6, 0.5),
            "waist_joint": np.zeros(3), "left_leg": np.zeros(6), "right_leg": np.zeros(6),
            "base_rot": np.array([0, 0, -1, 0.1, 0.2, 0.3]),
        },
        "base_pose": np.array([10, 20, 0.8, 0, 0, 0, 1]),
    }


@pytest.fixture
def statistics():
    return {"UnifoLM_WBT": {kind: {"offset": [0]*size, "scale": [1]*size}
                           for kind, size in (("action", 54), ("state", 60))}}


@pytest.fixture
def calibration():
    return json.loads((Path(__file__).parents[1] / "calibration.example.json").read_text())
