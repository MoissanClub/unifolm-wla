"""Capture latency is measured, not an arbitrary rejection threshold."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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

    def wait_for_frames(self):
        delay, value = next(self.arrivals)
        self.clock.now += delay
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
    sensor.stamps = {}
    sensor.feedback_ages = lambda: {}
    sensor.base_pose = lambda: np.array([1, 2, .8, 0, 0, 0, 1])
    sensor.kin = SimpleNamespace(fk=lambda motors, side: np.eye(4))
    low = SimpleNamespace(mode_machine=4, imu_state=SimpleNamespace(
        quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]))
    sensor.feedback = lambda: (low, np.zeros(35), {s: np.zeros(6) for s in robot.ARM_IDS})
    return sensor


@pytest.mark.parametrize("delay", [.5, 6.0])
def test_slow_capture_is_reported_and_queued_frame_is_discarded(clock, delay):
    pipeline = Pipeline(clock, [(delay, 7)], queued=[Frame(99)])
    capture_start = clock.now
    observation = sensors(pipeline).observe("pick up the can")
    assert observation["captured_monotonic"] == capture_start
    assert np.all(observation["images"]["head_left"] == 7)
    assert observation["capture_ms"] == pytest.approx(delay*1000)
    np.testing.assert_allclose(observation["state"]["left_ee_pose"][:3], [1, 2, .8])
    np.testing.assert_allclose(observation["state"]["left_fig6d"], np.ones(6))


def test_initial_missing_lowstate_is_retried_without_dds(monkeypatch, clock):
    low = SimpleNamespace(mode_machine=4)
    odom = SimpleNamespace(position=[0, 0, 0])
    hand = SimpleNamespace(states=[SimpleNamespace(q=0.) for _ in range(6)])
    subscriptions = {}
    initialized = []

    class Subscriber:
        def __init__(self, topic, message_type):
            self.topic = topic
            self.reads = 0
            self.closed = False
            subscriptions[topic] = self

        def Init(self, handler=None):
            if self.topic == "rt/lowstate":
                assert handler is None
            elif self.topic == "rt/odommodestate":
                handler(odom)
            else:
                handler(hand)

        def Read(self):
            self.reads += 1
            return None if self.reads == 1 else low

        def Close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", SimpleNamespace(
        ChannelFactoryInitialize=lambda *args: initialized.append(args),
        ChannelSubscriber=Subscriber))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_hg.msg.dds_",
                        SimpleNamespace(LowState_=object))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_go.msg.dds_",
                        SimpleNamespace(MotorStates_=object, SportModeState_=object))
    previous_reads = 0

    def wait_for_retry(_):
        nonlocal previous_reads
        # Fail deterministically if initialization waits without another read.
        # This detects the original infinite loop without a wall-clock timeout.
        reads = subscriptions["rt/lowstate"].reads
        assert reads > previous_reads
        previous_reads = reads

    monkeypatch.setattr(robot.time, "sleep", wait_for_retry)
    sensor = robot.Sensors("test-interface", object(), external_images=True)
    try:
        assert initialized == [(0, "test-interface")]
        assert subscriptions["rt/lowstate"].reads == 2
        assert sensor.low is low
        assert sensor.odom is odom
        assert set(sensor.hands) == {"left", "right"}
    finally:
        sensor.close()
    assert all(sub.closed for sub in subscriptions.values())


def test_standing_reference_keeps_floor_height_and_initial_heading():
    sensor = robot.Sensors.__new__(robot.Sensors)
    sensor.lock = robot.threading.Lock()
    sensor.reference = None
    initial_position = np.array([4., -2., -.3])
    initial_rotation = Rotation.from_euler("xyz", [.1, -.2, 1.2])

    def odometry(position, rotation, height):
        return SimpleNamespace(position=position, body_height=height,
                               imu_state=SimpleNamespace(quaternion=rotation.as_quat()[[3, 0, 1, 2]]))

    sensor.odom = odometry(initial_position, initial_rotation, .78)
    first = sensor.base_pose()
    np.testing.assert_allclose(first[:3], [0, 0, .78], atol=1e-14)
    np.testing.assert_allclose(Rotation.from_quat(first[3:]).as_matrix(),
                               Rotation.from_euler("xyz", [.1, -.2, 0]).as_matrix(), atol=1e-14)

    # The reference stays fixed after initialization, including when odometry
    # position and the firmware's body-height field subsequently change.
    reference = sensor.reference.copy()
    displacement = np.array([1., 2., -.1])
    new_rotation = Rotation.from_euler("xyz", [.1, -.2, 1.45])
    sensor.odom = odometry(initial_position + displacement, new_rotation, .91)
    second = sensor.base_pose()
    heading_inverse = Rotation.from_euler("z", -1.2)
    expected_position = heading_inverse.apply(displacement) + [0, 0, .78]
    np.testing.assert_allclose(second[:3], expected_position, atol=1e-14)
    np.testing.assert_allclose(Rotation.from_quat(second[3:]).as_matrix(),
                               (heading_inverse * new_rotation).as_matrix(), atol=1e-14)
    np.testing.assert_array_equal(sensor.reference, reference)
