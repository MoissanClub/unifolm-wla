"""Verify the BrainCo wire contract without DDS or physical hardware."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.g1.hands import BrainCoHands, FINGER_ORDER


@pytest.fixture
def sdk(monkeypatch):
    publishers = {}

    class Publisher:
        def __init__(self, topic, message_type):
            self.messages = []
            self.initialized = False
            self.closed = False
            self.succeeds = True
            publishers[topic] = self

        def Init(self):
            self.initialized = True

        def Write(self, message):
            self.messages.append(message)
            return self.succeeds

        def Close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel",
                        SimpleNamespace(ChannelPublisher=Publisher))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.default",
                        SimpleNamespace(unitree_go_msg_dds__MotorCmd_=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_go.msg.dds_",
                        SimpleNamespace(MotorCmds_=SimpleNamespace))
    return publishers


def test_construction_and_close_never_send_commands(sdk):
    hands = BrainCoHands()
    assert len(sdk) == 2
    assert all(p.initialized and not p.messages for p in sdk.values())
    hands.close()
    hands.close()
    assert all(p.closed and not p.messages for p in sdk.values())


def test_left_write_preserves_wire_order_without_commanding_right(sdk):
    hands = BrainCoHands()
    positions = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]
    hands.write("left", positions)
    assert FINGER_ORDER == ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
    message, = sdk["rt/brainco/left/cmd"].messages
    assert [c.q for c in message.cmds] == positions
    assert [c.dq for c in message.cmds] == [1.0] * 6
    assert not sdk["rt/brainco/right/cmd"].messages
    hands.close()
    assert len(sdk["rt/brainco/left/cmd"].messages) == 1


@pytest.mark.parametrize("target", [
    [0.0] * 5,
    [0.0] * 7,
    [[0.0] * 6],
    [float("nan")] + [0.0] * 5,
    [float("inf")] + [0.0] * 5,
])
def test_invalid_target_never_publishes(sdk, target):
    hands = BrainCoHands()
    with pytest.raises(ValueError):
        hands.write("left", target)
    assert all(not p.messages for p in sdk.values())
    hands.close()


def test_invalid_side_and_closed_publisher_never_publish(sdk):
    hands = BrainCoHands()
    with pytest.raises(ValueError, match="side"):
        hands.write("both", np.zeros(6))
    hands.close()
    with pytest.raises(RuntimeError, match="closed"):
        hands.write("left", np.zeros(6))
    assert all(not p.messages for p in sdk.values())


def test_regression_overshoot_reaches_existing_hardware_bridge(sdk):
    hands = BrainCoHands()
    positions = [-0.001, 1.001, 0.2, 0.3, 0.4, 0.5]
    hands.write("left", positions)
    message, = sdk["rt/brainco/left/cmd"].messages
    assert [c.q for c in message.cmds] == positions
    hands.close()


def test_failed_dds_write_is_reported(sdk):
    hands = BrainCoHands()
    sdk["rt/brainco/left/cmd"].succeeds = False
    with pytest.raises(RuntimeError, match="DDS write failed"):
        hands.write("left", np.zeros(6))
    hands.close()
