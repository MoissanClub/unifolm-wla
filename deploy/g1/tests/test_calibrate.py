"""Offline calibration fits preserve geometry and report measurement disagreement."""
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deploy.g1.calibrate import fit_samples, main


def pose(xyz, angles):
    result = np.eye(4)
    result[:3, 3] = xyz
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    return result


def sample(side, wrist, offset):
    return {"side": side, "pelvis_T_wrist": wrist.tolist(),
            "pelvis_T_policy": (wrist @ offset).tolist()}


def test_recovers_both_offsets_from_different_wrist_poses():
    offsets = {"left": pose([.123456789012, -.024, .013], [.1, -.2, .3]),
               "right": pose([.112, .032, -.019], [-.3, .2, -.1])}
    wrists = [pose([.1, .2, .3], [0, 0, 0]), pose([.3, .1, .2], [.4, -.3, .2]),
              pose([.2, -.1, .4], [-.2, .5, -.1])]
    records = [sample(side, wrist, offset) for side, offset in offsets.items() for wrist in wrists]
    calibration, report = fit_samples(records)
    assert report["sample_count"] == 6
    assert report["missing_sides"] == []
    for side, offset in offsets.items():
        np.testing.assert_allclose(calibration["wrist_T_policy"][side], offset, atol=1e-12)
        assert report["sides"][side]["sample_count"] == 3
        for residual in report["sides"][side]["residuals"]:
            assert residual["position_error_m"] < 1e-12
            assert residual["rotation_error_rad"] < 1e-12


def test_one_sample_neither_invents_other_hand_nor_claims_validation():
    offset = pose([.18, .03, 0], [.4, 0, 0])
    calibration, report = fit_samples([sample("left", np.eye(4), offset)])
    np.testing.assert_allclose(calibration["wrist_T_policy"]["left"], offset, atol=1e-12)
    assert set(calibration["wrist_T_policy"]) == {"left"}
    assert report["missing_sides"] == ["right"]
    assert report["sides"]["left"]["sample_count"] == 1
    assert "verified" not in calibration and "validated" not in calibration
    assert "repeatability" in report["note"]


def test_disagreement_is_reported_without_an_acceptance_cutoff():
    records = [sample("left", np.eye(4), pose([0, 0, 0], [0, 0, 0])),
               sample("left", np.eye(4), pose([2, 0, 0], [0, 0, 1]))]
    calibration, report = fit_samples(records)
    fitted = np.asarray(calibration["wrist_T_policy"]["left"])
    np.testing.assert_allclose(fitted[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(fitted[:3, :3], Rotation.from_euler("z", .5).as_matrix(), atol=1e-12)
    for residual in report["sides"]["left"]["residuals"]:
        assert residual["position_error_m"] == pytest.approx(1)
        assert residual["rotation_error_rad"] == pytest.approx(.5)


def test_rotation_mean_handles_angle_wraparound():
    records = [sample("left", np.eye(4), pose([0, 0, 0], [0, 0, np.deg2rad(angle)]))
               for angle in (179, -179)]
    calibration, report = fit_samples(records)
    rotation = np.asarray(calibration["wrist_T_policy"]["left"])[:3, :3]
    np.testing.assert_allclose(rotation, Rotation.from_euler("z", np.pi).as_matrix(), atol=1e-12)
    for residual in report["sides"]["left"]["residuals"]:
        assert residual["rotation_error_rad"] == pytest.approx(np.deg2rad(1))


@pytest.mark.parametrize("invalid", [np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]),
                                     np.full((4, 4), np.nan), np.eye(3)])
def test_invalid_rigid_pose_is_rejected(invalid):
    record = sample("left", np.eye(4), np.eye(4))
    record["pelvis_T_policy"] = invalid.tolist()
    with pytest.raises(ValueError):
        fit_samples([record])


def test_cli_writes_fit_and_complete_measurement_evidence(tmp_path):
    source = tmp_path / "measurements.json"
    output = tmp_path / "calibration.json"
    data = {"notes": "Synthetic test data", "samples": [sample("left", np.eye(4), np.eye(4))]}
    source.write_text(json.dumps(data))
    main(["--samples", str(source), "--out", str(output)])
    calibration = json.loads(output.read_text())
    report = json.loads((tmp_path / "calibration.report.json").read_text())
    assert report["input"] == data
    assert report["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert calibration["source"] == report["source"]
    assert calibration["evidence_report"] == str(tmp_path / "calibration.report.json")


def test_empty_measurements_are_not_a_calibration():
    with pytest.raises(ValueError, match="nonempty"):
        fit_samples([])
