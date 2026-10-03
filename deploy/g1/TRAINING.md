# Demonstrations and fine-tuning

Use the same camera roles, calibrated EEF frames, BrainCo channel order and action profile for recording,
training and deployment. This guide starts with **stationary bimanual manipulation**, suitable for the
23-DoF robot once base placement makes the required motions reachable.

## Record actual teleoperation targets

`DemoWriter.add()` accepts a measured observation plus the commands being issued by the teleoperator.
It computes command EEF poses with the actual 23-DoF URDF, stores JPEGs and timestamped numeric records,
and rejects an overflowing recording queue. The exporter rejects timing gaps and uses a 30 Hz grid with
a maximum 20 ms nearest-frame error. Record phase success explicitly.

To integrate into the existing teleoperation process, put `unifolm-wla` on that process's `PYTHONPATH`.
After its SDK initialization and before recording, construct these objects:

```python
import json
from pathlib import Path
from deploy.g1.robot import Kinematics23, Sensors
from deploy.g1.demonstrations import DemoWriter

calibration = json.loads(Path("/path/to/measured-g1-calibration.json").read_text())
kin = Kinematics23("/home/unitree/junda/xr_teleoperate/assets/g1/g1_body23.urdf", calibration)
observer = Sensors("enP8p1s0", kin, external_images=True, initialize_dds=False)
writer = DemoWriter("/data/fridge-demos/open-door/session01-episode001", kin)
```

Replace the example PC2 home directory with its real workspace path. In the 30 Hz control/record loop,
where `teleop_hand_and_arm.py` already has fresh head/wrist frames, five-joint arm state/command splits
and BrainCo hand targets, call:

```python
import cv2

images = {"head_left": cv2.cvtColor(head_img.bgr, cv2.COLOR_BGR2RGB)}
# If the head input is stereo side-by-side, select the left image before conversion.
if left_wrist_img is not None:
    images["cam_wrist_left"] = cv2.cvtColor(left_wrist_img.bgr, cv2.COLOR_BGR2RGB)
if right_wrist_img is not None:
    images["cam_wrist_right"] = cv2.cvtColor(right_wrist_img.bgr, cv2.COLOR_BGR2RGB)
obs = observer.observe("Hold the door with the right hand and take the can with the left hand", images=images)
writer.add(obs,
           arm_targets={"left": left_arm_action, "right": right_arm_action},
           hand_targets={"left": left_hand_action, "right": right_hand_action})
```

Use the instruction for the phase actually being demonstrated. The hand targets must be six values in
`[0,1]`, in bridge order `[thumb_close, thumb_lateral, index, middle, ring, little]`; convert telemetry
units explicitly if your controller exposes another representation. Use the calibrated **same** EEF
frame for measured and commanded poses. Verify image freshness in the camera owner's timestamps:
passing an old shared image into `observe()` does not make it fresh.

At the end of each episode:

```python
writer.close(success=True)  # use the actual observed result
# Start another DemoWriter for the next episode. At process exit:
observer.close()
```

This hook intentionally reuses the existing image owner instead of opening the RealSense twice.
It subscribes to measured legs, waist and IMU for model conditioning; the recorded **action** targets
contain arms and hands only. It cannot train simultaneous stepping/pulling. Do not label measured
future joint positions as if they were the commands, or use model predictions as successful expert labels.

The old XR `EpisodeWriter` records joint targets and images but lacks the synchronized WBT state fields
and capture timestamps required here. Existing `g1-fridge-fetch` event/image logs likewise are not a
ready-to-train WLA dataset. Keep the new complete records.

## Convert and select train/validation sessions

On the workstation, activate a separate, provisioned conda training environment with LeRobot 0.5.0
and the upstream training/video dependencies (see the deployment guide's environment requirements):

```bash
conda activate g1wla-train
python -m deploy.g1.demonstrations /data/fridge-demos/train \
  --out /data/g1-train/UnifoLM_WBT_Dataset/fridge_skills
python -m deploy.g1.demonstrations /data/fridge-demos/validation \
  --out /data/g1-validation/UnifoLM_WBT_Dataset/fridge_skills
python -m deploy.g1.prepare_data --data-root /data/g1-train --cache /data/g1-cache \
  --stationary-arms --head-only --out playground/g1-train.yaml
python -m deploy.g1.prepare_data --data-root /data/g1-validation --cache /data/g1-cache-validation \
  --stationary-arms --head-only --out playground/g1-validation.yaml
```

`train/` and `validation/` each contain episode directories directly; group different phases into those
directories with their per-frame task instructions. Omit `--head-only` when all demonstrations and
deployment have wrist cameras. The converter saves only complete episodes marked successful; failures
need separate recovery annotations before becoming training examples. It writes LeRobot v3 metadata,
Parquet and H.264 video, using the exact semantic state/action columns expected by the generated config.

The generated config retains the upstream normalization files. For an initial adaptation this preserves
the pretrained representation. Check new-state/action ranges against these stats. If recomputing stats,
use training sessions only and recompute **relative** action statistics for the same 30-frame horizon.
Do not silently substitute validation-derived normalization. The deployed bundle must include the
resulting `dataset_statistics.json` emitted by the training run.

## Train and merge

```bash
bash deploy/g1/finetune.sh \
  playground/Pretrained_models/UnifoLM-WLA-1.0-Base \
  playground/g1-train.yaml playground/Checkpoints/g1-fridge-lora lora
```

Use `full` instead of `lora` for full action-head tuning. The wrapper defaults to a 200-step pilot with
20 warmup steps. After validating that pilot, set, for example, `MAX_TRAIN_STEPS=20000 WARMUP_STEPS=200`
before the command, using a new run directory; choose the actual budget from validation. The wrapper
references the existing `ds_config.yaml` rather than the missing file in upstream's launchers. It refuses
to reuse an existing run directory. No cloud account, paid instance or job is launched by this delivery.

Inspect trainable parameters: the current `freeze_modules: qwen_vl_interface` also freezes the nested
state projector. If adaptation requires projector training, change the freezing policy explicitly and
verify actual trainable names and memory. Do not assume the prose description matches the code.

Package a **full** checkpoint, including any trained non-LoRA parameters:

```bash
python -m deploy.g1.finalize \
  --checkpoint playground/Checkpoints/g1-fridge-lora/final_model/model.safetensors \
  --run-dir playground/Checkpoints/g1-fridge-lora \
  --base-dir playground/Pretrained_models/UnifoLM-WLA-1.0-Base \
  --out playground/Pretrained_models/g1-fridge-lora-merged
python -m deploy.g1.evaluate \
  --checkpoint playground/Pretrained_models/g1-fridge-lora-merged/checkpoints/model.safetensors \
  --data-config playground/g1-validation.yaml --out results/g1-fridge-validation
```

Merging uses CPU RAM; allow at least 32 GB and enough disk for another full checkpoint. The formatter
validates checkpoint keys and merges injected LoRA layers; adapter-only checkpoints are rejected.
The conversion and training commands need validation with real collected episodes and a real trained
checkpoint; no such artifacts were available in this session.

Deploy the merged stationary policy with `server --profile stationary_arms`, retaining the matching
camera setup. Before contact execution, implement and validate the contact controller and phase-completion
checks described in [FRIDGE_PLAN.md](FRIDGE_PLAN.md). The included free-space arm test does not gain
contact-control or safe handover capabilities merely because its weights were fine-tuned.
