from pathlib import Path
import numpy as np
import pytest

from deploy.g1.robot import Kinematics23, ARM_IDS, transform


@pytest.fixture
def kin(calibration):
    pytest.importorskip("pinocchio")
    urdf = Path(__file__).resolve().parents[4] / "xr_teleoperate/assets/g1/g1_body23.urdf"
    return Kinematics23(urdf, calibration)


@pytest.mark.parametrize("side", ["left", "right"])
def test_actual_23dof_fk_ik_roundtrip(kin, side):
    q = np.zeros(35)
    q[list(ARM_IDS[side])] = [-.2, .2 if side == "left" else -.2, 0, 1, .1]
    target = kin.fk(q, side)
    solved = kin.ik(q, side, target)
    test_q = q.copy()
    test_q[list(ARM_IDS[side])] = solved
    np.testing.assert_allclose(kin.fk(test_q, side), target, atol=1e-5)


def test_unreachable_target(kin):
    target = np.eye(4)
    target[:3, 3] = [3, 0, 3]
    with pytest.raises(ValueError, match="unreachable"):
        kin.ik(np.zeros(35), "left", target)


def test_calibration_rejects_reflection():
    bad = np.eye(4)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="reflection"):
        transform(bad)
