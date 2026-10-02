# Findings, checked 2026-09-30

The released model is a useful starting point for an evaluation/fine-tuning project. The evidence does
not support “download, convert to TensorRT, and fetch a drink” on this 23-DoF G1.

## Evidence reviewed

Read both repository READMEs, the action/state/statistics specification, both action-expert guides,
third-party attributions, model loading and inference code, the
MMDiT action head, action/state mapping, SE(3) math, dataset schemas/normalization, evaluation scripts,
training/LoRA code and launch configs. Cross-checked the official project page, current Hugging Face
model metadata, checkpoint tensor header, and BrainCo fridge/shelf dataset metadata. Reviewed the local
G1 joint/arm/camera/API references, BrainCo bridge, teleoperation recording format and existing fridge
controller. The action/state specification is present in English and Chinese and agrees with the core
pose/frame conventions used here. The training guide's linked LoRA smoke-test files and launcher target
`deepseeds/deepspeed_zero2.yaml` are absent; the existing `ds_config.yaml` is used instead.

| Artifact | Verified identifier / fact |
|---|---|
| GitHub checkout and current upstream HEAD | `0a1aa87be8f775c7883fcd27216e3666ac18a8c5`, Sep 29 |
| HF base model revision | `dedc0612c4a921446db3a8cb0864bb7f7433329e` |
| Checkpoint file | `checkpoints/model.safetensors`, 12,458,898,964 bytes |
| Checkpoint tensor header | 1,256 tensors; 6,229,360,182 parameters; all BF16 |
| Qwen/VLM/projector group | 4,838,880,768 parameters, 9,677,761,536 bytes |
| Action expert group | 1,390,479,414 parameters, 2,780,958,828 bytes |
| Statistics keys | `UnifoLM_G1_Dex1`, `UnifoLM_WBT` |
| Action model config | DiT-L, 16 layers, 30 actions/chunk, four flow steps, 54 output slots |
| State input | 60 values plus a 60-value availability mask into a 120-input state projector |

These are observations from public files, not benchmark results. The model card itself contains almost
no deployment guidance. Sources: [repository](https://github.com/unitreerobotics/unifolm-wla),
[model files](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base/tree/dedc0612c4a921446db3a8cb0864bb7f7433329e),
[checkpoint config](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base/blob/dedc0612c4a921446db3a8cb0864bb7f7433329e/config.yaml).

## What the published system demonstrates

Unitree reports roughly 2,500 hours of robot training and demonstrations across 64 tasks: 10 whole-body
and 54 tabletop. The whole-body gallery labels cover trash disposal, laundry loading, shoe organization,
bathroom/kitchen tidying, bed making, collecting vegetables, cabinet filing, shelf placement and sofa
tidying. Tabletop examples include folding cloth, stacking cups/blocks/bowls, organizing cola, packaging,
microwave interaction and cable unplugging. These are publisher demonstrations, not success-rate guarantees
for the downloadable checkpoint on arbitrary G1 variants.
[Official project and video gallery](https://unigen-x.github.io/unifolm-wla.github.io/).

The published material reviewed does not provide a complete per-task success table with trials, seeds,
hardware/controller versions and deployment clients sufficient to reproduce every demonstration.
Video labels do not establish that every task generalizes to a new home, fridge, camera or arm geometry.
The ER/ER-Flow models are perception/reasoning backbones; the WLA base checkpoint includes the continuous
action expert. They are not interchangeable deployment artifacts.

## Capability relevant to this task

| Requirement | Released evidence | What must be added or evaluated |
|---|---|---|
| Turn until fridge enters view | Whole-body data and relative/base outputs exist | Search policy, stable detection, localization, obstacle handling and stop criteria |
| Approach and stop | Locomotion-related outputs exist | G1 controller integration, depth-based standoff, drift/occlusion handling |
| Open door while retaining control with one hand | Cabinet/appliance interaction is adjacent evidence | Your handle, hinge, seal force, arm reach, contact-compliant controller and bimanual coordination |
| Retrieve a can with other hand | Pick-up-drink and fridge-placement datasets exist | Reverse direction is not automatic; train retrieval, occlusion and retained-door-contact states |
| Close while holding drink | No reviewed evidence for this complete combination | Collision-aware closure, latch verification and preservation of the can grasp |
| Return to requester | No reviewed end-to-end benchmark | Localization/map or verified return route; identify the requester |
| Handover and release | No reviewed benchmark for your setup | Receiving-hand/contact confirmation; retain object on timeout |

Relevant primary datasets:
[BrainCo put drinks into fridge](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Put_Drinks_Into_Fridge),
[BrainCo pick up drinks](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Pick_Up_Drinks),
[BrainCo shelf organizing](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Supermarket_Shelf_Organizing),
[WBT collection](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset).

The fridge repository's `meta/info.json` reports LeRobot **v3.0**, 200 episodes, 127,423 frames, 30 Hz,
seven-joint arm arrays, six-channel BrainCo fingers and head/left/right wrist video. The Hugging Face
landing page displays 33 video rows; those rows are not the episode count. The shelf metadata likewise
reports 200 episodes. Use pinned files and `meta/info.json`, not viewer row counts, for data accounting.
AV1 is used in the public videos; test the workstation video decoder before a long download/training run.

## Why the stock server cannot drive this robot

1. `_ACTIVE_SLOT_SPECS` and the live websocket server expose Dex1 grippers, not BrainCo fingers or WBT
   base-pose actions. A separate legacy `active_rot6d_to_wbc50` references slots not in its active layout.
   The new server has a separate, explicit WBT contract; it does not merely rename the gripper channels.
2. The **state** slot named `base_rotvec` now holds body projected gravity and angular velocity, whereas
   the **action** slot holds a relative SE(3) base pose. A stale adapter docstring suggests otherwise.
   Putting odometry XYZ into the state slot changes the model input meaning.
3. Arm action translations are local to the observation's end-effector frame. Every future pose is
   `T_current @ T_relative[i]`, not accumulated deltas and not XYZ addition in the pelvis frame.
4. Your 23-DoF body has five joints per arm and yaw-only waist. The examined public datasets describe
   seven-joint arms and yaw/roll/pitch waist. Full 6D targets are not generally attainable on a five-joint
   arm. Missing joints must not be made into real motors by zero-padding a 29-DoF command.
5. WLA is a low-rate action predictor, not a stabilizing joint controller. The released repository does
   not supply a validated 23-DoF WBC client/controller for these outputs. Sending its leg angles directly
   to `rt/lowcmd` would not recreate the demonstration controller.
6. Current dataset configuration uses head and wrist cameras. The downward-facing RealSense-only view
   is a distribution shift. Missing wrist images are not fixed by copying the head view into three slots.
7. The upstream server's default whole-model BF16 conversion meets a head that creates FP32 masked
   actions under FP32 CUDA autocast (which disables autocast). The deployment wrapper keeps head linear
   operations under BF16 autocast and returns NumPy FP32, avoiding the mixed-dtype path.
8. Importing the action mapper previously imported the entire training dataloader stack. The only core
   package changes here make those imports lazy; PC2's sensor client does not require LeRobot or torch.

## Jetson and conversion

JetPack 6.2 corresponds to L4T 36.4.3, CUDA 12.6 and TensorRT 10.3 in NVIDIA's release notes; installation
alone does not prove that a particular Python environment can import them. PyTorch wheels must match
JetPack/Python. The current work session is on **x86_64 with CPU PyTorch and no accessible NVIDIA GPU**,
not on PC2. Board memory and TensorRT installation are workspace/user facts, not measurements made here.
[JetPack 6.2 release notes](https://docs.nvidia.com/jetson/jetpack/6.2/release-notes/index.html),
[NVIDIA PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

Safetensors is a weight container. An ONNX export specifies operations; a TensorRT engine compiles those
operations for a target runtime/GPU. No official full-policy TensorRT engine/exporter is in this pinned
repository. Qwen multimodal packing, rotary positions, the robot-state embedding hook, variable token
counts and the iterative action network make “convert the model file” an engineering project.
[NVIDIA TensorRT workflow](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html).

The experimental action-head split can reduce part of inference time but leaves about **78% of saved
parameter bytes** in the VLM group. FP16/BF16 conversion does not materially reduce weight memory.
VLM INT8/INT4 quantization, feature caching, or distillation are possible subsequent projects, requiring
held-out action/pose-error validation; none is claimed implemented or validated here. Stock text-generation
serving is not a drop-in replacement for the VLM hidden-state and robot-state-token interface.

## Fine-tuning facts and caveats

Upstream supports full action-head tuning and LoRA. Its default full-head example targets a single 24 GB
GPU, batch size one, gradient accumulation eight, 20,000 steps and learning rate 1e-4. That is a published
starting configuration, not a measured memory promise. Start a short pilot before allocating a long run.
[Training guide](https://github.com/unitreerobotics/unifolm-wla/blob/0a1aa87be8f775c7883fcd27216e3666ac18a8c5/docs/train_action_expert_en.md).

Capacity inference from the verified parameter counts: frozen BF16 VLM weights plus a fully trainable
action head's BF16 weights/gradients and two FP32 Adam moment arrays already total about **24.55 GiB**,
before activations or optimizer master weights. With an FP32 master copy, that becomes about **29.73 GiB**.
Thus, the published single-24-GB full-head configuration needs a measured memory check; reducing batch
size alone does not eliminate this parameter/optimizer cost. Start with LoRA, or provision more memory,
offloading or distributed optimizer sharding for full-head training. These estimates are not measured peaks.

The code freezes all parameters beneath `qwen_vl_interface`, including its nested robot-state projector,
despite prose saying the projector trains. LoRA is injected only into selected backbone modules; other
action encoders/decoders may remain trainable. Thus an **adapter-only file can omit changed non-LoRA
weights**. `finalize.py` uses the full fine-tuned checkpoint and merges LoRA layers before packaging.
It also fixes the inference bundle layout: upstream's `final_model/` is not the expected `checkpoints/`
layout by itself. Keep the run's exact statistics and the custom tokenizer with the model.

The loader samples concatenated sources proportionally to their frame counts; configured source `weight`
is not applied. Do not promise a rehearsal ratio just by changing that value. Use an explicitly selected
episode corpus or implement/test a sampler. Existing aggregate statistics can be retained for an initial
fine-tune; recomputation must include observation-anchored relative EE actions, the exact 30-step horizon,
state rotations, gravity/omega conventions and training-only data. Editing only `meta/stats.json` does
not update this loader's `precollected_stats_path`.

## Validation status

New CPU tests cover action/state layout, normalization-key failures, SE(3) anchors, camera-role omission,
serialization, stale/invalid/unsupported motion rejection, actual 23-DoF URDF FK/IK, data configuration,
the split MMDiT velocity rollout against its upstream implementation, actual command recording, LoRA
merging with non-adapter updates, and small-model ONNX export/reference-runtime parity. All **34 tests
passed** on CPU (Python 3.10.21, PyTorch 2.14.0+cpu, NumPy 1.26.4; Transformers 5.5.3 and
Diffusers 0.35.2). Model imports, Python/shell syntax and targeted Ruff checks also passed.

Instantiating the complete model on PyTorch's meta device against the downloaded tokenizer/config
matched **all 1,256 checkpoint tensor names and shapes**, with no missing or unexpected keys and no
full parameter allocation. This checks architecture compatibility, not pretrained numerical behavior.

The pre-existing `g1-fridge-fetch` suite returned **14 passed, 3 failed**:

- Default can target IK position error 0.0242 m exceeds its 0.020 m threshold.
- Full synthetic task reaches search/approach, then fails opening the door: right hand target
  `[0.366, 0.137, 0.114]` has about 0.052 m position error.
- A stale camera-view assertion expects a ceiling below 1 m at 0.4 m; current calibration computes
  about 1.10 m. Do not weaken the physical reach checks to make this suite green.

No full model weight inference, Jetson latency, full-size ONNX/TRT export or hardware task success is claimed.
No changes were made to the existing fridge controller to hide these failures.
