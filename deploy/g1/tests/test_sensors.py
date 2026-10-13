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


@pytest.fixture
def mock_dds(monkeypatch):
    sdk = SimpleNamespace(subscribers=[], low=SimpleNamespace(), init_failure=None)
    hand = SimpleNamespace(states=[SimpleNamespace(q=0.) for _ in range(6)])

    class Subscriber:
        def __init__(self, topic, message_type):
            self.topic = topic
            self.closed = 0
            sdk.subscribers.append(self)

        def Init(self, handler=None):
            if self.topic == sdk.init_failure:
                raise RuntimeError("DDS initialization failed")
            if self.topic == "rt/odommodestate":
                handler(SimpleNamespace())
            elif self.topic.startswith("rt/brainco/"):
                handler(hand)

        def Read(self):
            return sdk.low

        def Close(self):
            self.closed += 1

    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", SimpleNamespace(
        ChannelFactoryInitialize=lambda *args: None, ChannelSubscriber=Subscriber))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_hg.msg.dds_",
                        SimpleNamespace(LowState_=object))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_go.msg.dds_",
                        SimpleNamespace(MotorStates_=object, SportModeState_=object))
    return sdk


def test_cancel_during_discovery_closes_created_subscribers(monkeypatch, mock_dds):
    mock_dds.low = None

    def cancel(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(robot.time, "sleep", cancel)
    with pytest.raises(KeyboardInterrupt):
        robot.Sensors("test-interface", object(), external_images=True)
    assert len(mock_dds.subscribers) == 4
    assert all(sub.closed == 1 for sub in mock_dds.subscribers)


def test_failed_dds_initialization_closes_partially_initialized_subscriber(mock_dds):
    mock_dds.init_failure = "rt/brainco/right/state"
    with pytest.raises(RuntimeError, match="DDS initialization failed"):
        robot.Sensors("test-interface", object(), external_images=True)
    assert len(mock_dds.subscribers) == 4
    assert all(sub.closed == 1 for sub in mock_dds.subscribers)


@pytest.fixture
def mock_cameras(monkeypatch):
    hardware = SimpleNamespace(cameras=[], starts=0, stops=0,
                               open_error=None, release_error=None,
                               start_error=None, stop_error=None)

    class RealSense:
        def start(self, config):
            hardware.starts += 1
            if hardware.start_error is not None:
                raise hardware.start_error

        def stop(self):
            hardware.stops += 1
            if hardware.stop_error is not None:
                raise hardware.stop_error

    class Camera:
        def __init__(self, path):
            self.path = path
            self.releases = 0
            hardware.cameras.append(self)

        def isOpened(self):
            if self.path == "right":
                if hardware.open_error is not None:
                    raise hardware.open_error
                return False
            return True

        def release(self):
            self.releases += 1
            if self.path == "left" and hardware.release_error is not None:
                raise hardware.release_error

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=Camera))
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        pipeline=RealSense, config=lambda: SimpleNamespace(enable_stream=lambda *args: None),
        stream=SimpleNamespace(color=0), format=SimpleNamespace(rgb8=0)))
    return hardware


@pytest.mark.parametrize("cancel", [False, True])
def test_partial_camera_startup_closes_all_resources(mock_dds, mock_cameras, cancel):
    if cancel:
        mock_cameras.open_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt if cancel else RuntimeError):
        robot.Sensors("test-interface", object(), wrist_left="left", wrist_right="right")
    assert mock_cameras.starts == mock_cameras.stops == 1
    assert len(mock_cameras.cameras) == 2
    assert all(camera.releases == 1 for camera in mock_cameras.cameras)
    assert all(sub.closed == 1 for sub in mock_dds.subscribers)


def test_startup_error_survives_cleanup_error_and_other_resources_close(mock_dds, mock_cameras):
    failure = KeyboardInterrupt()
    mock_cameras.open_error = failure
    mock_cameras.release_error = RuntimeError("Camera release failed")
    with pytest.raises(KeyboardInterrupt) as caught:
        robot.Sensors("test-interface", object(), wrist_left="left", wrist_right="right")
    assert caught.value is failure
    assert str(caught.value.__cause__) == "Camera release failed"
    assert mock_cameras.stops == 1
    assert all(camera.releases == 1 for camera in mock_cameras.cameras)
    assert all(sub.closed == 1 for sub in mock_dds.subscribers)


def test_cancel_inside_realsense_start_closes_subscribers(mock_dds, mock_cameras):
    mock_cameras.start_error = KeyboardInterrupt()
    mock_cameras.stop_error = RuntimeError("Pipeline did not start")
    with pytest.raises(KeyboardInterrupt):
        robot.Sensors("test-interface", object())
    assert mock_cameras.starts == mock_cameras.stops == 1
    assert all(sub.closed == 1 for sub in mock_dds.subscribers)


def test_running_camera_stop_error_does_not_skip_subscriber_cleanup(mock_dds, mock_cameras):
    sensor = robot.Sensors("test-interface", object())
    mock_cameras.stop_error = RuntimeError("Running camera failed to stop")
    with pytest.raises(RuntimeError, match="Running camera failed to stop"):
        sensor.close()
    assert all(sub.closed == 1 for sub in mock_dds.subscribers)
    sensor.close()
    assert mock_cameras.stops == 1


def test_external_images_return_retains_subscribers_until_idempotent_close(mock_dds):
    sensor = robot.Sensors("test-interface", object(), external_images=True)
    assert all(sub.closed == 0 for sub in mock_dds.subscribers)
    sensor.close()
    sensor.close()
    assert all(sub.closed == 1 for sub in mock_dds.subscribers)
