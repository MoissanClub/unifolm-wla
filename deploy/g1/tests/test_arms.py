"""Arm/hand scheduling and service ownership with an entirely fake Unitree SDK."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.g1 import arms
from deploy.g1.robot import ARM_IDS


@pytest.fixture
def sdk(monkeypatch):
    state = SimpleNamespace(publishers={}, status=0, switches=[], fsm=500,
                            switch_statuses={False: [], True: []}, list_status=None)

    class Publisher:
        def __init__(self, topic, message_type):
            self.messages = []
            self.closed = False
            state.publishers[topic] = self

        def Init(self):
            pass

        def Write(self, message):
            self.messages.append(copy.deepcopy(message))
            return True

        def Close(self):
            self.closed = True

    class Loco:
        def Init(self):
            pass

        def GetFsmId(self):
            return 0, state.fsm

    class Services:
        def Init(self):
            pass

        def ServiceList(self):
            status = state.status if state.list_status is None else state.list_status
            return 0, [SimpleNamespace(name=arms.ARM_SERVICE, status=status)]

        def ServiceSwitch(self, name, enabled):
            state.switches.append((name, enabled))
            pending = state.switch_statuses[enabled]
            state.status = pending.pop(0) if pending else 0 if enabled else 1
            return 0

        def _Call(self, api_id, parameter):
            assert api_id == 1001
            request = json.loads(parameter)
            code = self.ServiceSwitch(request["name"], bool(request["switch"]))
            return code, json.dumps({"name": request["name"], "status": state.status})

    def lowcmd():
        return SimpleNamespace(motor_cmd=[SimpleNamespace(q=0, dq=0, tau=0, kp=0, kd=0)
                                          for _ in range(35)])

    modules = {
        "unitree_sdk2py.core.channel": SimpleNamespace(ChannelPublisher=Publisher),
        "unitree_sdk2py.g1.loco.g1_loco_client": SimpleNamespace(LocoClient=Loco),
        "unitree_sdk2py.go2.robot_state.robot_state_client": SimpleNamespace(RobotStateClient=Services),
        "unitree_sdk2py.go2.robot_state.robot_state_api": SimpleNamespace(ROBOT_STATE_API_ID_SERVICE_SWITCH=1001),
        "unitree_sdk2py.idl.default": SimpleNamespace(
            unitree_hg_msg_dds__LowCmd_=lowcmd, unitree_go_msg_dds__MotorCmd_=SimpleNamespace),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": SimpleNamespace(LowCmd_=SimpleNamespace),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": SimpleNamespace(MotorCmds_=SimpleNamespace),
        "unitree_sdk2py.utils.crc": SimpleNamespace(CRC=lambda: SimpleNamespace(Crc=lambda msg: 0)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("DISCOVERY_S", "RAMP_IN_S", "RAMP_OUT_S"):
        monkeypatch.setattr(arms, name, 0.0)
    return state


@pytest.fixture
def sensors():
    motors = np.zeros(35)
    motors[list(ARM_IDS["right"])] = [0.1, 0.2, 0.3, 0.4, 0.5]
    motors[12] = 0.2
    return SimpleNamespace(feedback=lambda: (SimpleNamespace(mode_machine=4), motors.copy(), {}))


@pytest.fixture
def manual_session(sdk, sensors, monkeypatch):
    """Run the real publisher path on a controlled clock without a worker race."""
    session = arms.ArmSession(sensors)
    session.start()
    session.stop_event.set()
    session.thread.join()
    monkeypatch.setattr(session, "check", lambda: None)
    clock = SimpleNamespace(now=0.0)

    def sleep(duration):
        clock.now += duration
        session._publish()

    monkeypatch.setattr(arms, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    try:
        yield session, clock
    finally:
        session.close()


def test_constructor_and_unstarted_close_do_not_touch_robot(sdk, sensors):
    session = arms.ArmSession(sensors)
    session.close()
    session.close()
    assert not sdk.publishers
    assert not sdk.switches


@pytest.mark.parametrize("initial_status", [0, 1])
def test_service_ownership_restores_previously_reported_state(sdk, sensors, initial_status):
    sdk.status = initial_status
    session = arms.ArmSession(sensors)
    session.start()
    assert sdk.status == 1
    session.close()
    assert sdk.status == initial_status
    assert sdk.switches == ([(arms.ARM_SERVICE, False), (arms.ARM_SERVICE, True)]
                            if initial_status == 0 else [])
    assert all(p.closed for p in sdk.publishers.values())
    assert not sdk.publishers["rt/brainco/left/cmd"].messages
    assert not sdk.publishers["rt/brainco/right/cmd"].messages
    assert sdk.publishers["rt/arm_sdk"].messages[-1].motor_cmd[29].q == 0.0


def test_switch_response_confirms_ownership_despite_unchanged_service_list(sdk, sensors, capsys):
    sdk.list_status = 0
    session = arms.ArmSession(sensors)
    session.start()
    assert sdk.status == 1
    assert session._service_status() == 0
    session.close()
    assert sdk.status == 0
    confirmations = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(entry["enabled"], entry["switch_status"]) for entry in confirmations] == [(False, 1), (True, 0)]


def test_pending_switches_retry_acquisition_and_restoration(sdk, sensors):
    sdk.switch_statuses = {False: [0, 0, 1], True: [1, 0]}
    session = arms.ArmSession(sensors)
    session.start()
    assert sdk.status == 1
    session.close()
    assert sdk.status == 0
    assert [enabled for _, enabled in sdk.switches] == [False, False, False, True, True]


def test_interrupt_pending_acquisition_restores_service_without_publishers(sdk, sensors, monkeypatch):
    sdk.switch_statuses = {False: [0, 1], True: [1, 0]}
    sleeps = []

    def sleep(duration):
        sleeps.append(duration)
        if len(sleeps) == 2:  # Initial discovery, then a pending switch response.
            sdk.status = 1  # The backend finishes switching after its response.
            raise KeyboardInterrupt

    monkeypatch.setattr(arms, "time", SimpleNamespace(monotonic=arms.time.monotonic, sleep=sleep))
    session = arms.ArmSession(sensors)
    with pytest.raises(KeyboardInterrupt):
        session.start()
    assert sdk.status == 0
    assert [enabled for _, enabled in sdk.switches] == [False, True, True]
    assert not sdk.publishers
    assert session.closed


@pytest.mark.parametrize("code,response,reason", [
    (3104, {}, "3104"),
    (0, {"name": arms.ARM_SERVICE, "status": 5}, "protected"),
    (0, {"name": arms.ARM_SERVICE, "status": 7}, "unknown status"),
    (0, {"name": "other", "status": 1}, "Invalid arm-service switch response"),
    (0, {"name": arms.ARM_SERVICE}, "unknown status"),
])
def test_switch_errors_are_not_retried(sdk, sensors, code, response, reason):
    calls = []

    def call(api_id, parameter):
        calls.append((api_id, parameter))
        return code, json.dumps(response)

    session = arms.ArmSession(sensors)
    session.services = SimpleNamespace(_Call=call)
    with pytest.raises(RuntimeError, match=reason):
        session._switch_service(False)
    assert len(calls) == 1


def test_nonstanding_mode_never_takes_authority(sdk, sensors):
    sdk.fsm = 0
    session = arms.ArmSession(sensors)
    with pytest.raises(RuntimeError, match="FSM 500"):
        session.start()
    session.close()
    assert not sdk.publishers
    assert not sdk.switches


@pytest.mark.parametrize("failed_topic", ["rt/arm_sdk", "rt/brainco/left/cmd"])
def test_cleanup_failure_still_restores_service_and_closes_other_publishers(sdk, sensors, failed_topic):
    session = arms.ArmSession(sensors)
    session.start()

    def fail():
        raise RuntimeError("DDS close failure")

    sdk.publishers[failed_topic].Close = fail
    with pytest.raises(RuntimeError, match="DDS close failure"):
        session.close()
    assert sdk.status == 0
    assert sdk.switches[-1] == (arms.ARM_SERVICE, True)
    assert all(p.closed for topic, p in sdk.publishers.items() if topic != failed_topic)
    assert not session.thread.is_alive()


def test_interpolation_starts_measured_and_uses_all_frames():
    points = np.array([[0.0] * 5, [0.4] * 5, [0.8] * 5])
    np.testing.assert_allclose(arms.interpolate(points, 0, 30), points[0])
    np.testing.assert_allclose(arms.interpolate(points, 1 / 60, 30), [0.2] * 5)
    np.testing.assert_allclose(arms.interpolate(points, 1 / 30, 30), points[1])
    np.testing.assert_allclose(arms.interpolate(points, 2 / 30, 30), points[2])
    np.testing.assert_allclose(arms.interpolate(points, 10, 30), points[2])


def test_left_plan_holds_right_waist_and_does_not_open_hand_on_close(sdk, sensors):
    session = arms.ArmSession(sensors)
    session.start()
    plan = {"side": "left", "fps": 30, "joints": np.array([[0.2] * 5, [0.4] * 5]),
            "hands": np.array([[0.1] * 6, [0.6] * 6])}
    try:
        session.play(plan)
        np.testing.assert_allclose(session.command["left"], plan["joints"][-1])
        assert session.trajectory is None
    finally:
        session.close()
    hand_messages = sdk.publishers["rt/brainco/left/cmd"].messages
    targets = [[c.q for c in m.cmds] for m in hand_messages]
    changes = [target for index, target in enumerate(targets)
               if index == 0 or target != targets[index - 1]]
    assert changes == plan["hands"].tolist()
    assert not sdk.publishers["rt/brainco/right/cmd"].messages
    _, motors, _ = sensors.feedback()
    for message in sdk.publishers["rt/arm_sdk"].messages:
        assert message.motor_cmd[12].q == motors[12]
        assert [message.motor_cmd[i].q for i in ARM_IDS["right"]] == motors[list(ARM_IDS["right"])].tolist()
    last = sdk.publishers["rt/arm_sdk"].messages[-1]
    assert [last.motor_cmd[i].q for i in ARM_IDS["left"]] == plan["joints"][-1].tolist()


def test_all_hand_keyframes_keep_timing_then_refresh_through_inference(manual_session, sdk, monkeypatch):
    session, clock = manual_session
    sends = []
    write = session.hands.write

    def record(side, target):
        sends.append((clock.now, np.asarray(target).copy()))
        return write(side, target)

    monkeypatch.setattr(session.hands, "write", record)
    # Standing before playback must not invent a hand target.
    clock.now = 1.0
    session._publish()
    assert not sends
    plan = {"side": "left", "fps": 30, "joints": np.zeros((30, 5)),
            "hands": np.linspace(0, 1, 30)[:, None] * np.ones((30, 6))}
    started = clock.now
    session.play(plan)
    np.testing.assert_allclose([stamp for stamp, _ in sends], started + np.arange(1, 31) / 30)
    np.testing.assert_allclose([target for _, target in sends], plan["hands"])
    # No new chunk is supplied during this simulated two-second inference.
    for _ in range(20):
        clock.now += 0.101
        session._publish()
    assert len(sends) == 50
    np.testing.assert_allclose([target for _, target in sends[30:]],
                               np.repeat(plan["hands"][-1][None], 20, axis=0))
    assert not sdk.publishers["rt/brainco/right/cmd"].messages


def test_close_stops_overdue_heartbeat_in_final_publish(manual_session, sdk):
    session, clock = manual_session
    session.play({"joints": np.zeros((1, 5)), "hands": np.full((1, 6), 0.7)})
    before = len(sdk.publishers["rt/brainco/left/cmd"].messages)
    clock.now += 1.0
    session.close()
    assert len(sdk.publishers["rt/brainco/left/cmd"].messages) == before
    assert not sdk.publishers["rt/brainco/right/cmd"].messages
    assert sdk.publishers["rt/arm_sdk"].messages[-1].motor_cmd[29].q == 0.0


def test_failed_hand_heartbeat_is_reported_by_arm_loop(manual_session, sdk, monkeypatch):
    session, clock = manual_session
    session.play({"joints": np.zeros((1, 5)), "hands": np.full((1, 6), 0.7)})
    monkeypatch.setattr(sdk.publishers["rt/brainco/left/cmd"], "Write", lambda message: False)
    clock.now += 1.0
    session.stop_event.clear()
    session._loop()
    assert session.stop_event.is_set()
    assert session.fault == "BrainCo left DDS write failed"


def test_invalid_plan_never_sends_hand_commands(sdk, sensors):
    session = arms.ArmSession(sensors)
    session.start()
    try:
        with pytest.raises(ValueError, match="schema"):
            session.play({"joints": np.zeros((3, 5)), "hands": np.zeros((2, 6))})
    finally:
        session.close()
    assert not sdk.publishers["rt/brainco/left/cmd"].messages


def test_changed_fsm_stops_next_chunk_without_hand_commands(sdk, sensors):
    session = arms.ArmSession(sensors)
    session.start()
    sdk.fsm = 1
    try:
        with pytest.raises(RuntimeError, match="FSM 500"):
            session.play({"joints": np.zeros((2, 5)), "hands": np.zeros((2, 6))})
    finally:
        session.close()
    assert not sdk.publishers["rt/brainco/left/cmd"].messages
