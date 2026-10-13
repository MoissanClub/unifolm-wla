"""Explicit BrainCo six-finger commands through the existing DDS bridge.

ChannelFactoryInitialize must be called by the coordinator. Construction and
shutdown send no targets; the bridge retains its last commanded grasp.
"""
from __future__ import annotations

import numpy as np


FINGER_ORDER = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")


class BrainCoHands:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

        self._motor_command = unitree_go_msg_dds__MotorCmd_
        self._message = MotorCmds_
        self.publishers = {}
        try:
            for side in ("left", "right"):
                publisher = ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
                self.publishers[side] = publisher
                publisher.Init()
        except BaseException:
            self.close()
            raise

    def write(self, side, target):
        """Send q in FINGER_ORDER, using the SDK's full normalized speed (dq=1).

        The existing BrainCo bridge enforces its native hardware position range.
        No additional regression-output cutoff is applied here. No movement is
        sent to the other hand.
        """
        if side not in ("left", "right"):
            raise ValueError("Hand side must be left or right")
        q = np.asarray(target, dtype=float)
        if q.shape != (len(FINGER_ORDER),) or not np.all(np.isfinite(q)):
            raise ValueError("BrainCo target must contain six finite positions")
        if side not in self.publishers:
            raise RuntimeError("BrainCo publishers are closed")
        message = self._message()
        message.cmds = [self._motor_command() for _ in FINGER_ORDER]
        for command, position in zip(message.cmds, q):
            command.q = float(position)
            command.dq = 1.0
        if self.publishers[side].Write(message) is False:
            raise RuntimeError(f"BrainCo {side} DDS write failed")

    def close(self):
        """Release DDS publishers without opening or otherwise moving either hand."""
        publishers, self.publishers = self.publishers, {}
        errors = []
        for publisher in publishers.values():
            try:
                publisher.Close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
