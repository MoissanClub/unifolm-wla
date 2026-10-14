# Supervised left-hand table pickup

This runbook evaluates the released WLA policy on the G1 Edu 23-DoF with standard BrainCo adapters
and the existing head-camera stream. It uses BrainCo's 20 cm wrist-roll reference as an explicit
geometry assumption; see [CALIBRATION.md](CALIBRATION.md). Five model-driven arm/hand chunks were executed on October 14, but a physical can pickup has not
yet been established. See the [trial and resume record](reports/2026-10-14-table-pickup.json). No measured 4×4 matrices are required for this initial experiment.

## Physical preparation

Place the can where it is visible in the head camera and within an ordinary left-arm reach. Leave
space for the hand and arm to approach it. Stay beside the G1 with its remote control. Use the
robot's normal operating procedure to enter a stable standing **FSM 500** before execution; FSM 0
is not the required standing mode. The client checks the mode but does not change it. No particular
table height, can distance, or artificial starting pose is required by this client.

Software checks and camera verification can be handled remotely. Complete those checks with the
model server stopped first. The commands below run on G1 PC2, in the existing conda
environment; they install no packages. The RTSP stream must already be available at
`rtsp://127.0.0.1:8554/d435`. This path consumes the existing camera stream instead of opening the
RealSense device directly.

## 1. Check the live observation without loading the model

```bash
cd ~/unifolm-wla
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate g1fetch
python -m deploy.g1.client --record-only --steps 1 \
  --head-stream rtsp://127.0.0.1:8554/d435 \
  --calibration deploy/g1/calibration.brainco23.reference.json \
  --instruction "pick up the can with the left hand"
```

Inspect the saved head image and observation: the can should be visible, feedback should correspond
to the current robot, and the assumed floor/base height should be plausible. This command sends no
arm or hand commands. The reference file supplies both hand poses because the policy observes both
hands even when only the left hand is selected for execution.

## 2. Start the model in a separate terminal

After the observation and software checks finish:

```bash
cd ~/unifolm-wla
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate g1fetch
python -m deploy.g1.server \
  --checkpoint playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --allow-missing-cameras
```

Wait for the server to finish loading. Keep this terminal running. `--allow-missing-cameras` permits
head-only inference; it does not synthesize wrist views or make this equivalent to three-camera
evaluation. Starting the model server does not activate robot motion.

## 3. Preview one prediction without motion

In the client terminal with `g1fetch` still active:

```bash
python -m deploy.g1.client --plan --side left --steps 1 \
  --head-stream rtsp://127.0.0.1:8554/d435 \
  --calibration deploy/g1/calibration.brainco23.reference.json \
  --instruction "pick up the can with the left hand"
```

This saves the observation, policy prediction, and all planned joint/hand targets under the printed
`results/g1/...` directory. Inspect the planned reach and hand commands, IK position/orientation
residuals, and unexecuted whole-body outputs. There is no added numerical acceptance cutoff. A
feasible-looking plan is diagnostic evidence, not proof that a grasp will succeed.

## 4. Execute one supervised chunk

With the G1 standing in FSM 500, the can still in view, and the operator beside it with the remote:

```bash
python -m deploy.g1.client --execute --side left --steps 1 \
  --head-stream rtsp://127.0.0.1:8554/d435 \
  --calibration deploy/g1/calibration.brainco23.reference.json \
  --instruction "pick up the can with the left hand"
```

This captures a fresh observation and predicts again; it does not replay the earlier preview. It
executes one complete action chunk, which may only start the reach. `--steps 1` is the operator's
experiment selection, not a hardcoded limit; the client has no maximum step count.

The selected arm and hand execute the model trajectory. The other arm and waist retain their initial
targets, and the standing controller retains leg and balance control. While waiting for another
prediction in a continuing run, the arm holds its last target. Inspect whether the physical motion
and hand behavior correspond to the plan before continuing.

Arm targets stream at 50 Hz on `rt/arm_sdk`; left-hand finger commands use the separate
`rt/brainco/left/cmd` topic. Before taking arm authority, the client checks the reported
`g1_arm_example` service state and disables that service when reported running. Switches are
confirmed from the API 1001 `ServiceSwitch` response (`0` running, `1` stopped), following the
[Unitree example](../../../unitree_sdk2/example/go2/go2_robot_state_client.cpp). PC2's `ServiceList`
can retain an old status after a successful switch; its initial value supplies the previously
reported state for release/restore decisions, not confirmation of a switch.

**Normal completion and Ctrl-C release arm authority** through the SDK weight ramp and restore the
previously reported built-in arm-service state. The client does not open the hand, but it also does
not maintain the final arm pose indefinitely after exit. Arm behavior then belongs to the
standing/built-in controller; account for this when the hand is near the table or holding the can. For an immediate
physical stop, use the operator's existing remote stop procedure.

## 5. Continue closed-loop evaluation

Once the initial physical behavior has been checked, omit `--steps` to keep observing, predicting,
and executing until Ctrl-C:

```bash
python -m deploy.g1.client --execute --side left \
  --head-stream rtsp://127.0.0.1:8554/d435 \
  --calibration deploy/g1/calibration.brainco23.reference.json \
  --instruction "pick up the can with the left hand"
```

Watch the reach, finger closure, and lift. There is no automatic pickup-success detector or automatic
stop after grasping. A completed chunk or successful DDS write does not establish task success.
Save the printed run directory and describe the observed outcome when comparing trials. Ctrl-C in
the separate server terminal stops the model and releases its GPU memory after the client has exited.

## How to interpret unsuccessful trials

The 5-DoF arm cannot reproduce arbitrary six-dimensional targets, and this executor does not apply
the policy's base, leg, or waist actions. The head-only input differs from the released camera setup.
Onboard inference has taken about 1.8 seconds per request, so execution holds between chunks and
plays each complete chunk after inference; this differs from the training timing. The reference
hand geometry and firmware-derived floor/base height also remain assumptions to examine if reaches
are consistently displaced.

These differences prevent a guarantee of zero-shot pickup. First use the saved observations and
plans to distinguish an input/frame/IK problem from an unsuccessful policy prediction. If repeated
trials remain unsuccessful after those checks, collect demonstrations on this 23-DoF robot with the
same cameras and stationary-arm action profile, then follow [TRAINING.md](TRAINING.md) to fine-tune
and evaluate on held-out trials. Fine-tuning is a possible next step, not an established prerequisite
for trying the released policy.
