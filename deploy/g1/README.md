# UniFoLM-WLA on the G1 Edu 23-DoF / BrainCo

This is an **experimental evaluation package**, not a validated autonomous fridge-fetch deployment.
It adds the missing BrainCo WBT data path, model serving, robot observation capture, offline evaluation,
stationary arm/hand evaluation, demonstration conversion, fine-tuning launch/packaging, and an
experimental TensorRT action-head path. No robot motion was performed while developing it.

Start with [research and limitations](RESEARCH.md), [the complete fridge task plan](FRIDGE_PLAN.md),
and [demonstration collection and training](TRAINING.md).
The [restriction audit](LIMITS.md) lists the removed deployment cutoffs and remaining hardware/protocol
requirements and control settings.

The [October 5 onboard deployment report](DEPLOYMENT.md) records a verified download, preserved
conda dependencies, and full BF16 inference on the 16 GB Orin NX without robot motion.
Warm synthetic inference took 1.63 s with one view and 3.49 s with three views at 15 W.

## What runs where

| Component | Recommended first evaluation | Entirely onboard experiment |
|---|---|---|
| Cameras, DDS, FK/IK, arm test | PC2, existing robot Python environment | Same |
| Released 6.23B policy | Separate NVIDIA GPU, preferably 24 GB or more | Offline inference verified; exceeds the action horizon at 15 W |
| Fine-tuning / ONNX export | Workstation GPU / sufficiently large host | Not the initial PC2 workflow |
| TensorRT engine build | Destination Jetson, same TRT version | Same |

Weights alone occupy **11.60 GiB** at BF16. The Orin NX's memory is shared with the OS, cameras,
CUDA workspaces and activations. The full model fits this 16 GB machine with little remaining memory;
measured inference at 15 W exceeds the one-second action horizon. An 8 GB module cannot hold the
full BF16 weights. The loader avoids constructing a second full copy of the model. The current
experimental executor holds position during inference and then plays a complete chunk from index zero;
this stop-and-act timing is different from a real-time whole-body controller.

All commands below run from `unifolm-wla/`. `python` means the activated environment's interpreter.
Use the existing conda environment. Installing the upstream project's full dependency set into PC2's
robot environment would select different torch, NumPy and training packages.

## 1. Install and inspect

On PC2, reuse the proven `g1fetch` conda environment and its activation hooks:

```bash
cd ~/junda/unifolm-wla
CONDA_ENV=g1fetch bash deploy/g1/setup.sh client --check
CONDA_ENV=g1fetch bash deploy/g1/setup.sh client
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate g1fetch
python -m deploy.g1.assets preflight
```

`--check` is an offline inventory/import check and exits nonzero for missing or incompatible dependencies.
The normal command installs missing packages only. `--dry-run` resolves and displays proposed additions
without installing them; it may access the package index. Reruns skip pip when requirements are satisfied.

Setup freezes **every installed Python distribution**, including transitive dependencies. It checks pip's
proposed transaction before applying the exact hashed archives with `--no-deps`. Existing incompatible
versions cause a stop, not an upgrade or downgrade. It also refuses to install native/GPU packages through
the application dependency resolver: torch/torchvision, CUDA/cuDSS, TensorRT, NumPy/SciPy, Pinocchio,
OpenCV, RealSense and DDS must already be provisioned. NumPy stays below 2 for the existing Pinocchio ABI.
Model-library pins remain the tested versions; a different installed version is reported for review.

`CONDA_ENV` accepts an existing environment name or prefix; it defaults to `g1fetch`. Set `CONDA_EXE`
if conda is outside PATH and `~/miniconda3`. Setup activates conda so CUDA/cuDSS library hooks take effect.
It does not create environments, run apt, rewrite activation hooks or replace system TensorRT bindings.
If you need isolation, explicitly clone the working environment first, for example
`conda create -n g1wla --clone g1fetch --offline`, then use `CONDA_ENV=g1wla`.

The [PC2 deployment learnings](../../../g1-fridge-fetch/docs/05_deploy_learnings.md) and
[dependency details](../../../g1-fridge-fetch/docs/06_dependency_learnings.md) are the platform reference:
Python 3.10, the Jetson `jp6/cu126` torch build (verified there: torch 2.11.0 / torchvision 0.26.0),
cuDSS library paths, and system TensorRT 10.3 exposed to conda. Model setup tests an actual BF16 CUDA
matrix multiply and imports TensorRT on Jetson. A missing library or wrong torch build stops setup;
repair the identified component using those notes before retrying. No camera or DDS connection is opened.

On the inference machine (PC2 for the onboard experiment, otherwise the GPU host):

```bash
CONDA_ENV=g1fetch bash deploy/g1/setup.sh model --check
CONDA_ENV=g1fetch bash deploy/g1/setup.sh model
conda activate g1fetch
python -m deploy.g1.assets fetch
python -m deploy.g1.assets inspect
```

The download is pinned to HF revision `dedc0612c4a921446db3a8cb0864bb7f7433329e`.
`fetch --metadata-only` omits the 12.46 GB weight file. The default bundle directory is
`playground/Pretrained_models/UnifoLM-WLA-1.0-Base`.
The Python 3.10 client and model imports were tested here; upstream's reference environment is Python 3.12.
CUDA execution and the dependency-preserving Jetson installation passed on October 5; see the
[onboard report](DEPLOYMENT.md) for versions, measurements, and remaining limitations.

## 2. Start the policy server

```bash
python -m deploy.g1.server \
  --checkpoint playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors
```

Default: loopback HTTP/msgpack on port 8601, BF16 PyTorch, SDPA, WBT statistics and three camera roles.
This is a separate protocol from the upstream Dex1 websocket server on 8600. Do not connect a Dex1/WBC
client to it. It loads no robot SDK and sends no robot commands.

For a robot with only the head camera, explicitly add `--allow-missing-cameras`. This is a camera
ablation; it is not equivalent to the released three-view evaluation. Missing cameras are omitted with
their role labels; the head view is never duplicated to impersonate wrist views.

For a remote inference host, keep the server on loopback and open this tunnel **from PC2**:

```bash
ssh -N -L 8601:127.0.0.1:8601 USER@GPU_HOST
```

The protocol sends uncompressed RGB arrays, so use a reliable LAN and measure complete round-trip latency.
Do not expose this unauthenticated research server directly to the Internet.

## 3. Capture or evaluate PC2 observations without motion

The process needs sole access to the selected cameras. Inspect the running camera service and stop it
when you are ready to switch ownership; this package does not stop services automatically.
Capture duration, feedback age and inference latency are reported as measurements, without added
time cutoffs. Each observation discards the queued RealSense frame and waits for another frame.
Make a calibration copy:

```bash
cp deploy/g1/calibration.example.json /tmp/g1-calibration.json
python -m deploy.g1.client --record-only --steps 10 \
  --calibration /tmp/g1-calibration.json --instruction "pick up the can with the left hand"
```

The example wrist-to-policy transforms are identity **placeholders**. The model's `*_gripper_base`
end-effector frames must be measured relative to your wrist-roll joints. Until then, shadow predictions
are diagnostic only. The default URDF is your sibling `xr_teleoperate/assets/g1/g1_body23.urdf`.

To query the running policy and save its predictions:

```bash
python -m deploy.g1.client --steps 20 --calibration /tmp/g1-calibration.json \
  --instruction "pick up the can with the left hand"
```

Add `--wrist-left /dev/v4l/by-id/LEFT_DEVICE --wrist-right /dev/v4l/by-id/RIGHT_DEVICE` for wrist cameras.
Default DDS interface: `enP8p1s0`. Every run saves its calibration, server metadata, observations,
predictions and timing beneath `results/g1/`. Omitting `--steps` runs until Ctrl-C.
The sensor path polls `rt/lowstate` and subscribes to odometry and the two BrainCo state topics.
It does not build the existing fridge `Robot` object,
whose construction starts publishers and changes the arm service.

The released WBT dataset uses `fig6d = 1 - native BrainCo q` and end-effector poses in a world
reference frame, including the floating base. These relationships were verified numerically across
22,585 released rows; see [the evidence](reports/2026-10-13-dataset-semantics.json).
The live client creates a standing task reference with initial x/y/yaw zero and floor height from
the firmware's `body_height`; raw odometry z has an arbitrary startup origin on this PC2.
Old observations recorded before these corrections are not valid live-policy evaluation inputs.
The corrected wire protocol is `unifolm-g1-wbt-v2`; restart both client and server after updating.
The firmware height is an assumed floor reference, not an independently measured pelvis height.

Replay a saved observation on the inference host:

```bash
python -m deploy.g1.benchmark \
  --checkpoint playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --observation /path/to/00000.observation.msgpack --out results/benchmark-01 --iterations 10 --save-probe
```

This separates cold/warm inference, VLM/head time, token count and CUDA allocation. On PC2 also record
`tegrastats` for total shared-memory use, temperature, throttling and power mode; PyTorch allocation alone
does not account for all Jetson memory.

## 4. Evaluate recorded Unitree episodes

Use a **separate, provisioned workstation conda training environment** with upstream's LeRobot 0.5.0
and video/training dependencies. This deployment installer covers the client, model server and ONNX
export; it does not provision the full training stack. The upstream Python 3.12 training dependency set
belongs in that separate environment, not the robot's Python 3.10 environment:

```bash
conda activate g1wla-train
export DATA_ROOT="$PWD/playground/g1-data"
hf download unitreerobotics/G1_WBT_Brainco_Put_Drinks_Into_Fridge --repo-type dataset \
  --revision 7c6a9f562d599a84186ae866198a2e06ee6d9ad5 \
  --local-dir "$DATA_ROOT/UnifoLM_WBT_Dataset/G1_WBT_Brainco_Put_Drinks_Into_Fridge"
python -m deploy.g1.prepare_data --data-root "$DATA_ROOT" --cache "$PWD/playground/g1-cache" \
  --out playground/g1-wbt.yaml
python -m deploy.g1.evaluate \
  --checkpoint playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --data-config playground/g1-wbt.yaml --episode 0 --out results/fridge-episode-00
```

The generated config uses the upstream WBT schema and exact `UnifoLM_WBT` normalization key. The evaluator
fails on statistics mismatches rather than silently choosing Dex1. It records per-hand position,
rotation and finger errors plus normalized MAE and latency. Repeating with different episode IDs tests
more observations. Action agreement on likely pretraining data is a plumbing baseline, not evidence
of held-out task success. Keep independently collected evaluation episodes out of training.

## 5. Preview and explicit arm/hand execution

Preview the complete selected-arm trajectory without publishing commands:

```bash
python -m deploy.g1.client --plan --side left --steps 1 \
  --calibration /path/to/measured-g1-calibration.json \
  --instruction "pick up the can with the left hand"
```

Plans contain all action frames, five-joint IK targets, native BrainCo targets, IK residuals and
unexecuted whole-body outputs. There is no age-based prefix skipping or rejection based on fixed
Cartesian/joint movement, base motion, hand separation or IK-error thresholds. The IK solver keeps
the physical URDF joint ranges. A 5-DoF arm generally cannot realize every 6D pose; inspect its residuals.

`--execute` explicitly enables the selected arm and hand using the supplied calibration. The former
`verified` boolean gate has been removed; setting a JSON flag never established physical calibration.
It requires FSM 500 and manages the competing built-in arm service. The robot holds position while
inference runs, then starts the complete chunk at index zero. This supports slow stop-and-act evaluation;
it does not reproduce Unitree's full-body WBT controller. The other arm and waist retain their initial
targets; the standing controller retains leg and balance control. Exit releases arm authority and
preserves hand closure.

After the physical frame has been established, the explicit execution command is:

```bash
python -m deploy.g1.client --execute --side left \
  --calibration /path/to/measured-g1-calibration.json \
  --instruction "pick up the can with the left hand"
```

Omit `--steps` for continuous observation/prediction/playback until Ctrl-C. There is no automatic
grasp-completion detector; a successful DDS write or completed chunk is not proof of a pickup.

The installed 23-DoF BrainCo mounting transform is still unmeasured. The example remains an identity
placeholder, and the 29-DoF dataset geometry is not proof of this robot's physical mounting.
`calibration.brainco23.reference.json` records BrainCo's published 20 cm teleoperation convention as an
unverified reference. Neither that file nor an identity transform establishes the WLA grasp frame.
No hardware can-pickup success has been established. The executor has no object-contact or
collision model and does not claim task success when a chunk finishes.

For a model specifically fine-tuned with the stationary-arm action mask, start the server with
`--profile stationary_arms`; match the training camera configuration too. Do not present this as the
base model's original WBT evaluation.

## 6. Experimental TensorRT path

No format conversion is needed for PyTorch: `.safetensors` already contains the weights. TensorRT needs
an exported computational graph plus an engine. This package exports **only the 1.39B action velocity
network**, leaving tokenization, image processing, Qwen3-VL, robot-state injection, flow integration
and pose decoding outside TensorRT. It is not a full-policy engine and does not quantize the VLM.

On a host with sufficient RAM (the FP32 ONNX weights alone are about 5.56 GB):

```bash
CONDA_ENV=g1wla bash deploy/g1/setup.sh export --check
CONDA_ENV=g1wla bash deploy/g1/setup.sh export
conda activate g1wla
python -m deploy.g1.trt \
  --checkpoint playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --out results/trt/action.onnx --tokens 1024
```

The example uses an existing `g1wla` export environment; change `CONDA_ENV` to your actual environment.
Export may use CPU PyTorch >=2.8. The same dependency preservation rules apply; an installed ONNX 1.23
is reused, and `ml-dtypes` 0.6 is rejected because it requires NumPy 2.

Copy the ONNX **and all external data files** to PC2, then build there:

```bash
bash deploy/g1/build_engine.sh results/trt/action.onnx results/trt/action.engine
python -m deploy.g1.trt_parity \
  --checkpoint playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --engine results/trt/action.engine --probe results/benchmark-01/probe.npz \
  --out results/trt/parity-01.json
```

Repeat parity over representative task/pose/camera inputs and measure physical decoded errors. The tool
reports errors without a default pass/fail threshold; supply `--max-abs` for an explicit tolerance.
Then benchmark using `--engine results/trt/action.engine`. The server
accepts the same option. CPU tests verify that the split velocity wrapper preserves the upstream flow
rollout and that a small instance exports to ONNX with numerical parity under ONNX's reference evaluator.
Full-size export, TRT parsing, precision parity and Jetson speed remain unverified here. Export/parser
errors are real blockers, not something a file extension change can solve. Build with JetPack 6.2's
TRT 10.3 tools; a workstation/TRT 11 engine is not a substitute.

## Verification and boundaries

Run `python -m pytest deploy/g1/tests -q` in an environment with the client dependencies, pytest,
Pinocchio, the model dependencies for the velocity test, and PEFT for the merge test.
Preserve the NumPy version used by compiled robotics packages when adding export dependencies;
NumPy 2 can be ABI-incompatible with existing Pinocchio binaries. These tests do not contact a robot.
The local validation report is in [RESEARCH.md](RESEARCH.md).

`fridge_baseline.sh sim` and `fridge_baseline.sh check` expose the separate structured controller for
comparison. The September 30 test results in RESEARCH.md predate its latest bring-up work; use the
[October 2 deployment learnings](../../../g1-fridge-fetch/docs/05_deploy_learnings.md) for its newer hardware
status. Those notes confirm search and approach, with door manipulation still unverified.
See [FRIDGE_PLAN.md](FRIDGE_PLAN.md) for the work needed to reach a defensible full task trial.
