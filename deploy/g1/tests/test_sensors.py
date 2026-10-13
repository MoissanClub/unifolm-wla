"""Camera startup and freshness regressions without a robot or camera."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from deploy.g1 import robot


class Clock:
    now = 100.0

    def monotonic(self):
        return self.now


class Frame:
    def __init__(self, value):
        self.value = value

    def get_color_frame(self):
        return self

    def get_data(self):
        return np.full((2, 2, 3), self.value, np.uint8)


class Pipeline:
    def __init__(self, clock, arrivals, queued=()):
        self.clock = clock
        self.arrivals = iter(arrivals)
        self.queued = list(queued)

    def wait_for_frames(self, timeout_ms):
        delay, value = next(self.arrivals)
        self.clock.now += min(delay, timeout_ms / 1000)
        if delay > timeout_ms / 1000:
            raise RuntimeError("Frame timeout")
        return Frame(value)

    def poll_for_frames(self):
        return self.queued.pop(0) if self.queued else None


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(robot.time, "monotonic", clock.monotonic)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    return clock


def sensors(pipeline):
    # Isolate capture from DDS and physical hardware. No publisher is involved.
    sensor = robot.Sensors.__new__(robot.Sensors)
    sensor.pipeline = pipeline
    sensor.cameras = {}
    sensor.kin = SimpleNamespace(fk=lambda motors, side: np.eye(4))
    low = SimpleNamespace(mode_machine=4, imu_state=SimpleNamespace(
        quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]))
    sensor.feedback = lambda: (low, np.zeros(35), {s: np.zeros(6) for s in robot.ARM_IDS})
    return sensor


def test_slow_startup_is_excluded_and_queued_frame_is_discarded(clock):
    pipeline = Pipeline(clock, [(0.5, 1)] + [(0.033, 1)] * 9 + [(0.033, 7)],
                        queued=[Frame(99)])
    robot._warmup_realsense(pipeline)
    capture_start = clock.now
    observation = sensors(pipeline).observe("pick up the can")
    assert observation["captured_monotonic"] == capture_start
    assert np.all(observation["images"]["head_left"] == 7)
    assert clock.now - capture_start < robot.CAMERA_ACQUISITION_S


def test_slow_live_capture_still_fails(clock):
    with pytest.raises(RuntimeError, match="Camera acquisition exceeds 200 ms"):
        sensors(Pipeline(clock, [(0.25, 7)])).observe("pick up the can")


def test_startup_cannot_wait_forever(clock):
    with pytest.raises(RuntimeError, match="timeout"):
        robot._warmup_realsense(Pipeline(clock, [(6.0, 1)]))
    assert clock.now == 105.0


def test_camera_queue_cannot_drain_forever(clock):
    pipeline = Pipeline(clock, [])

    def continually_queued():
        clock.now += 0.05
        return Frame(99)

    pipeline.poll_for_frames = continually_queued
    with pytest.raises(RuntimeError, match="queue did not drain"):
        sensors(pipeline).observe("pick up the can")
