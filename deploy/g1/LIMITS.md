# Deployment limits audit — 2026-10-13

This **local proposed revision** removes the deployment-specific timing and motion cutoffs below. It adds
complete selected-arm/BrainCo chunk planning and explicit execution. It does not establish
physical can-pickup success: no model-driven robot motion was performed for this revision.

The full patch has **not been deployed to G1**. Automatic approval review rejected the upload because
it would enable actuation with unmeasured calibration after removing motion cutoffs. The deployed
robot remains on the earlier code (`300fc61`); the protocol and behavior described here apply to the
proposed patch. The inference server was subsequently stopped at the user's request to free memory.

Old values were checked against commit `300fc61`, particularly `guard.py`, `robot.py`,
`arms.py`, `client.py`, and `calibration.example.json`. Numerical settings, hardware ranges,
and data-integrity checks still exist; this is not a claim that every number or condition vanished.

## Removed capture, transport, and input restrictions

| Previous restriction | Current behavior |
|---|---|
| Camera acquisition and queue draining: **200 ms** each | Capture time is recorded without an added cutoff. Discard a queued RealSense frame before waiting for another. |
| Camera warm-up: **10 frames within 5 s** | Removed this startup gate. |
| RealSense frame wait: explicit **1 s** | Use the SDK's own wait behavior; no deployment timeout override. |
| Initial robot/hand feedback wait: **5 s**; feedback freshness: **200 ms** | Wait for required streams; record feedback ages without those rejection thresholds. |
| DDS callback queues: **1 sample**; requested wrist-camera buffer: **1 frame** | Removed these overrides. Lowstate is polled on demand; the SDK/backend owns buffering. |
| Client request timeout: **30 s**; execution request deadline: remaining **0.45 s** | No application request timeout or action-age deadline. |
| Server socket timeout: **10 s** | Removed the override. |
| HTTP request and response ceilings: **24 MiB** | Read the declared request/full response; reject malformed or incomplete bodies. |
| Default client run: **10 observations** | Runs until Ctrl-C unless `--steps` explicitly supplies a count. |
| Image dimensions: height **32–1080**, width **32–1920** | Accept nonempty RGB arrays; inference still resizes to its configured model resolution. |
| Action chunk ceiling: **120 frames** | Accept any nonempty chunk with the model's action schema. |
| Observed fingers outside **−0.001…1.001**; demonstration commands outside **0…1** | Removed these deployment rejection tests; finite six-channel data is still required. |
| Gravity norm tolerance **0.1**; base quaternion norm tolerance **0.01** | Removed these cutoffs. A zero quaternion remains mathematically invalid. |

See [robot.py](robot.py), [client.py](client.py), [server.py](server.py), and
[contract.py](contract.py). Third-party SDK, operating-system, and device timeouts were
not removed from their implementations.

## Removed action and controller cutoffs

| Previous restriction | Current behavior |
|---|---|
| Observation age **0.45 s**, reserved move duration **0.30 s** | No stale-action rejection. Hold position during inference, then play the full chunk from index zero. |
| Index `ceil((age + 0.30) × 30)` had to fit a **30-frame** chunk | Removed age-based prefix skipping and its additional roughly **0.67 s** age ceiling. |
| Predicted base speed **0.05 m/s** or yaw speed **0.05 rad/s** | Report unused base outputs; do not reject the selected arm plan on these values. |
| Predicted base translation **0.02 m**, rotation **0.05 rad** | Same: diagnostics, not gates. |
| Predicted leg/waist change **0.08 rad**; measured lower-body change since capture **0.05 rad** | Removed these rejection thresholds. |
| End-effector translation **0.03 m**, rotation **0.15 rad**, arm-joint change **0.15 rad** | Removed per-target magnitude gates. |
| Minimum hand separation **0.18 m** | Removed this fixed-distance rejection. |
| Body roll/pitch **0.2 rad**, joint tracking error **0.15 rad** | Removed these deployment fault thresholds. |
| Command increment clipped to **±0.005 rad** per **100 Hz** cycle | Removed increment clipping; interpolate the planned trajectory on its playback clock. |
| Locomotion RPC timeout **200 ms** | Use SDK transport defaults and allow DDS discovery before requests. |
| Periodic FSM check every **0.5 s**, scheduler reset after **50 ms** lateness | Removed those timer policies. Check FSM at arm acquisition and before each chunk. |
| Shutdown thread wait ceiling **1 s** | Join the publisher thread without an added timeout. |
| IK joint margin **0.02 rad**, maximum **50** evaluations | Use physical URDF bounds and the solver's default convergence behavior. |
| IK position residual **15 mm**, rotation residual **0.15 rad**, or nonconvergence caused rejection | Record residuals and convergence for every frame; no deployment acceptance threshold. |
| Calibration had to contain `verified: true` | Removed the Boolean gate. The supplied transform is still used; a flag does not establish its physical accuracy. |
| Hand outputs were logged but never commanded | Explicit execution now publishes the selected hand's six native targets with the arm trajectory. |

The former `guard.py` and calibration `limits` object are removed. See
[trajectory.py](trajectory.py), [arms.py](arms.py), and [hands.py](hands.py).
This executor selects one arm/hand. It does not execute the model's walking, waist, or other-arm
outputs; the standing controller retains balance, while the arm publisher holds the other arm
and waist targets. Removing cutoffs does not supply collision/contact reasoning or success detection.

## Removed recording, evaluation, training, and export defaults

| Previous restriction | Current behavior |
|---|---|
| Recorder aborted at **8 unfinished writes** | Queue writes and record peak backlog; no added backlog rejection. |
| Conversion required **30 samples**, gaps ≤ **66.7 ms**, resampling error ≤ **20 ms** | Accept nonempty, finite, increasing timestamps; save gaps, errors, and repeated-frame diagnostics. |
| Only successful demonstrations converted | `--successful-only` is opt-in; conversion records episode success annotations. |
| Evaluation defaulted to the first **300 source frames**, stride **10** | Evaluate the complete selected episode at stride **1**; explicit sampling options remain. |
| Benchmark defaulted to **10 warm iterations** | `--iterations` is required; one cold measurement is recorded separately. |
| Training wrapper imposed **200 steps / 20 warm-up steps** | Use upstream configuration defaults, currently **20,000 / 200**; environment overrides remain available. |
| Training launcher fixed **1 machine** | `NUM_MACHINES` and `MACHINE_RANK` are configurable, alongside `NUM_PROCESSES`. |
| TRT export assumed **1024 context tokens** | Require explicit `--tokens`; the built engine's actual shape remains binding. |
| TRT build workspace ceiling **1024 MiB** | Use TensorRT defaults unless `TRT_WORKSPACE_MIB` is explicitly supplied. |
| TRT parity default pass/fail tolerance **0.05** | Report measured differences without a numerical acceptance verdict unless `--max-abs` is supplied. Nonfinite results remain errors. |
| Model download required **15 GiB** free before checking cached files | Removed this precheck; download/storage errors report actual failures. |
| Safetensors header ceiling **10 MiB** | Check header length against the actual file length. |

Removing application queue/time ceilings does not remove RAM, disk, or device capacity.
In particular, a recorder whose writer cannot keep up can accumulate a growing memory backlog.

## Remaining requirements and configuration

- **Hardware/controller:** physical URDF joint ranges; 23-DoF joint mapping; six BrainCo channels;
  valid PR feedback (`mode_pr=0`, `mode_machine` 1 or 4); standing FSM 500; exclusive built-in
  arm-service ownership; successful DDS writes. The existing BrainCo bridge and firmware retain
  their native position/actuator behavior.
- **Intent and data integrity:** `--execute` explicitly enables motion; record/replay do not execute.
  Correct protocol/frame IDs, finite values, nonempty images/chunks, positive rates, model array
  shapes, compatible checkpoint keys, and valid transforms remain necessary. Rigid-transform
  checks and dataset-statistics comparison use `atol=1e-5` with NumPy's relative tolerance.
  Existing run-directory rejection prevents accidental overwrite.
- **Model/runtime:** 54 action and 60 unified state dimensions; released 30 Hz action timing;
  RGB camera roles and normalization statistics; CUDA/BF16 serving; one inference worker/batch;
  fixed TensorRT engine shapes; full compatible checkpoints rather than unmerged adapters.
  Missing wrist cameras still require explicit `--allow-missing-cameras`.
- **Controller settings:** 50 Hz arm publication, 1 s discovery settling, 2 s authority acquisition,
  and 1 s release. Arm stiffness is 60, wrist-roll stiffness 40, damping 1.5; commanded arm velocity
  and feedforward torque are zero. BrainCo `dq=1` uses the bridge's full normalized speed.
  These are retained control parameters, not validated task-success thresholds.
- **Capture/recording settings:** head RGB 640×480 at 30 FPS; model images default to 336×448;
  one demonstration writer, JPEG quality 95, 30 FPS conversion. Loopback port 8601 remains the
  server default. These settings are not claims of unlimited throughput or arbitrary camera support.
- **Environment/training:** conda and installed-version preservation remain the user's requirements.
  Tested package pins/ranges and Jetson Python 3.10 ABI checks remain in the requirement files and
  [dependencies.py](dependencies.py). Upstream frozen-VLM training configurations still freeze the
  nested state projector. Process/machine counts default to one, but are configurable. The historical
  deployment launch example sets `OMP_NUM_THREADS=4`; it is not a model guard and remains adjustable.
- **Scope:** offline evaluation selects one dataset/task/episode at a time. The separate
  `fridge_baseline.sh` wrapper still exposes only `sim` and `check`; it is not this policy executor.

## Protocol v2 and calibration status

The protocol is now **`unifolm-g1-wbt-v2`**. Restart the server with the matching client;
old v1 observations are not interchangeable with corrected live inputs.

- Native BrainCo DDS uses 0=open, 1=closed; WBT uses `fig6d = 1 - native_q` in both observations
  and targets, preserving channel order.
- WBT end-effector poses include the floating-base pose in a world/reference frame. Live FK is
  composed into that frame; decoded world targets are transformed back into the pelvis frame for IK.
- The live standing reference zeros initial x/y/yaw and uses firmware `body_height` for its floor
  reference. Raw odometry z alone has an arbitrary startup origin on this PC2.

The hand and frame relationships were checked against **22,585 rows** of the released dataset;
see [the recorded evidence](reports/2026-10-13-dataset-semantics.json). The recovered 29-DoF dataset
geometry uses a 0.105 m tool offset plus a 0.005 m wrist-yaw-origin correction to the local XR URDF.
Neither result measures this robot's installed 23-DoF BrainCo mounting transform.

BrainCo's [pinned reference IK implementation](https://github.com/BrainCoTech/unitree-g1-brainco-hand/blob/1c9a703797048c7da1a4031d5c4b9ca79f2904f3/brainco_ws/src/control_py/control_py/unitree_robot/robot_arm_ik.py#L371-L384)
uses a **0.20 m** end-effector offset in its reference setup. That value is not a measurement of this
installed 23-DoF wrist-to-policy frame. The example calibration remains an identity placeholder;
removing its Boolean gate does not resolve that geometry. No physical pickup, contact-control,
collision-avoidance, or handover success is claimed by this revision.
