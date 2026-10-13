# Measuring the installed hand frame

`calibrate.py` fits a fixed wrist-to-policy transform from physical pose measurements. It runs
offline, initializes no DDS or model, and sends no robot commands. It does not discover or validate
the physical policy frame, assign a `verified` flag, or decide whether a fit is accurate enough.

For each measurement, record two 4×4 rigid transforms in the same pelvis frame:

- `pelvis_T_wrist`: pose of the URDF wrist-roll joint, obtained from synchronized encoder FK or an
  independent measurement. This is the joint frame, without a guessed hand extension.
- `pelvis_T_policy`: independently measured pose of the physical hand frame corresponding to the
  policy's end-effector frame. Define its origin and axes in the measurement notes. Do not generate
  this pose using the wrist offset you are trying to measure.

Translations are metres. Rotation matrices must be proper orthonormal rotations, with columns
giving the local axes in the pelvis frame. Both poses must describe the same stationary instant.
One pose pair determines a transform but cannot demonstrate repeatability. Capture several
distinct stationary arm poses to expose measurement inconsistency, and check additional poses
independently afterward. There is no minimum pose count beyond having a measurement.

## Input and command

Use one JSON object with a `samples` list. This example contains **synthetic format examples, not
measurements of the G1**:

```json
{
  "notes": "Describe the physical frame, measurement method, units, and setup here.",
  "samples": [
    {
      "id": "left-01",
      "side": "left",
      "pelvis_T_wrist": [[1,0,0,0.3],[0,1,0,0.2],[0,0,1,0.1],[0,0,0,1]],
      "pelvis_T_policy": [[1,0,0,0.423],[0,1,0,0.224],[0,0,1,0.113],[0,0,0,1]]
    }
  ]
}
```

Add records for the right side using the same format. The utility fits only supplied sides; it
never inserts an identity transform for a missing hand. The current sensor client requires
calibration entries for both hands because both poses are included in the model observation.

```bash
conda activate g1fetch
python -m deploy.g1.calibrate \
  --samples /path/to/measured-pose-pairs.json \
  --out /path/to/measured-g1-calibration.json
```

The command writes the calibration and a sibling `measured-g1-calibration.report.json`. The report
contains the complete input, source SHA-256, sample counts, missing sides, and each sample's position
and rotation residual in metres/radians. It computes each candidate as
`inverse(pelvis_T_wrist) @ pelvis_T_policy`, averages translations, and uses a chordal SO(3)
rotation mean. It reports disagreement without rejecting samples or applying an acceptance cutoff.
Small residuals show consistency; shared measurement bias or an incorrectly chosen frame can
still produce small residuals.

## Obtaining the physical measurements

With an accessible mounting drawing and identifiable wrist origin, measure the complete translation
and orientation of the selected hand frame relative to the wrist. A ruler measurement along one
axis alone does not determine a general rigid transform.

An alternative is a rigid marker whose transform to the selected hand frame is measured. With a
calibrated camera and synchronized encoders:

```text
pelvis_T_policy = pelvis_T_camera @ camera_T_marker @ marker_T_policy
wrist_T_policy = inverse(pelvis_T_wrist) @ pelvis_T_policy
```

Save the resulting measured `pelvis_T_policy` with the encoder-derived wrist pose. Camera
extrinsics, marker placement, hand-frame definition, and encoder timing all contribute error.
Changing the adapter or moving the marker invalidates the corresponding measurement geometry.

BrainCo's [23-DoF example](https://github.com/BrainCoTech/unitree-g1-brainco-hand/blob/1c9a703797048c7da1a4031d5c4b9ca79f2904f3/brainco_ws/src/control_py/control_py/unitree_robot/robot_arm_ik.py#L371-L384)
uses a 20 cm extension along wrist-roll x with identity rotation. That is a reference convention;
it does not measure this robot's adapter or establish correspondence to WLA's grasp frame. Likewise,
the recovered 29-DoF dataset offset applies after wrist yaw and cannot substitute for a measurement
of the installed 23-DoF assembly.
