import numpy as np
import pytest

from deploy.g1.contract import WBTAdapter, PROTOCOL
from deploy.g1.guard import choose_target


class FakeKinematics:
    def fk(self, motors, side):
        pose = np.eye(4)
        pose[:3, 3] = [.3, .25 if side == "left" else -.25, .2]
        return pose

    def ik(self, motors, side, target):
        return np.zeros(5)


def reply_for(observation, statistics):
    reply = WBTAdapter(statistics).decode(np.zeros((30, 54)), observation)
    reply.update(protocol=PROTOCOL, frame_id=observation["frame_id"])
    return reply


def test_skip_elapsed_prefix(observation, statistics, calibration):
    reply = reply_for(observation, statistics)
    targets, index = choose_target(reply, observation, np.zeros(35), FakeKinematics(), calibration["limits"], now=100.1)
    assert index == 12
    assert set(targets) == {"left", "right"}


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(frame_id="wrong"), "different observation"),
    (lambda r: r["base_command"].__setitem__((slice(None), 0), .2), "locomotion"),
    (lambda r: r["base_relative"].__setitem__((slice(None), 0), .3), "translation"),
    (lambda r: r["waist"].__setitem__((slice(None), 1), .3), "waist"),
    (lambda r: r["legs"].__setitem__((slice(None), 0), .3), "lower-body"),
    (lambda r: r["left_T"].__setitem__((slice(None), 0, 3), 1), "Cartesian"),
    (lambda r: r["right_hand"].__setitem__((0, 0), np.nan), "Invalid action|finite"),
])
def test_motion_rejection(observation, statistics, calibration, mutation, match):
    reply = reply_for(observation, statistics)
    mutation(reply)
    with pytest.raises(ValueError, match=match):
        choose_target(reply, observation, np.zeros(35), FakeKinematics(), calibration["limits"], now=100.1)


def test_stale_reply_rejected(observation, statistics, calibration):
    with pytest.raises(ValueError, match="Stale"):
        choose_target(reply_for(observation, statistics), observation, np.zeros(35), FakeKinematics(),
                      calibration["limits"], now=101)


def test_unsupported_profile(observation, statistics, calibration):
    reply = reply_for(observation, statistics)
    reply["action_profile"] = "unknown"
    with pytest.raises(ValueError, match="profile"):
        choose_target(reply, observation, np.zeros(35), FakeKinematics(), calibration["limits"], now=100.1)
