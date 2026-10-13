"""Fit wrist-to-policy transforms from recorded physical pose measurements.

Offline only: no DDS, camera, model, or robot control is initialized.
Residuals describe agreement between measurements, not physical validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import transform


def rigid_pose(value):
    # Reuse deployment's SE(3) schema check, but retain the original float64
    # values instead of its inference-oriented float32 return value.
    pose = np.asarray(value, dtype=np.float64)
    transform(pose)
    return pose


def fit_samples(samples):
    """Return a calibration and residual report without inventing missing sides."""
    if not isinstance(samples, list) or not samples:
        raise ValueError("Expected a nonempty list of measured pose samples")
    grouped = {}
    for index, sample in enumerate(samples):
        side = sample["side"]
        if side not in ("left", "right"):
            raise ValueError(f"Sample {index}: side must be left or right")
        wrist = rigid_pose(sample["pelvis_T_wrist"])
        policy = rigid_pose(sample["pelvis_T_policy"])
        relative = np.linalg.inv(wrist) @ policy
        grouped.setdefault(side, []).append((index, sample.get("id"), wrist, policy, relative))

    calibration = {"wrist_T_policy": {}}
    report = {
        "method": "translation arithmetic mean; SO(3) chordal rotation mean",
        "sample_count": len(samples),
        "missing_sides": [side for side in ("left", "right") if side not in grouped],
        "note": "Residuals measure consistency, not physical accuracy or policy-frame identity. "
                "A single sample has no independent repeatability evidence.",
        "sides": {},
    }
    for side, measurements in grouped.items():
        relative = np.stack([item[4] for item in measurements])
        fitted = np.eye(4)
        fitted[:3, 3] = relative[:, :3, 3].mean(axis=0)
        fitted[:3, :3] = Rotation.from_matrix(relative[:, :3, :3]).mean().as_matrix()
        calibration["wrist_T_policy"][side] = fitted.tolist()
        residuals = []
        for index, sample_id, wrist, measured, _ in measurements:
            predicted = wrist @ fitted
            position_error = np.linalg.norm(predicted[:3, 3] - measured[:3, 3])
            rotation_error = (Rotation.from_matrix(predicted[:3, :3]).inv()
                              * Rotation.from_matrix(measured[:3, :3])).magnitude()
            residuals.append({"sample_index": index, "id": sample_id,
                              "position_error_m": float(position_error),
                              "rotation_error_rad": float(rotation_error)})
        report["sides"][side] = {"sample_count": len(measurements), "residuals": residuals}
    return calibration, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True,
                        help="JSON object containing a samples list of measured pose pairs")
    parser.add_argument("--out", type=Path, required=True, help="Output calibration JSON path")
    args = parser.parse_args(argv)
    report_path = args.out.with_name(args.out.stem + ".report.json")
    if args.samples.resolve() in (args.out.resolve(), report_path.resolve()):
        parser.error("Output paths must not overwrite the input measurements")
    raw = args.samples.read_bytes()
    document = json.loads(raw)
    calibration, report = fit_samples(document["samples"])
    source = {"path": str(args.samples.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    report.update(source=source, input=document)
    calibration.update(source=source, evidence_report=str(report_path.resolve()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"calibration": str(args.out), "report": str(report_path),
                      "sample_count": report["sample_count"], "missing_sides": report["missing_sides"]}))


if __name__ == "__main__":
    main()
