# Onboard deployment — 2026-10-05 UTC

The released UniFoLM-WLA-1.0-Base runs on this G1's Orin NX in BF16 PyTorch.
Installation, checkpoint integrity, synthetic inference, and the loopback policy API passed.
No camera capture, DDS connection, controller activation, or robot motion was performed.
The operator reported the robot in zero torque mode.

This validates the runtime, not task success: inputs were black RGB images and synthetic states.
No physical actions were executed or scored.

## Installed configuration

- G1 Edu 23 DoF, BrainCo hands; Orin NX 16 GB, L4T 36.4.3.
- Repository: `/home/unitree/unifolm-wla`; deployment code commit `ab9a1a6`.
- Existing conda environment: `/home/unitree/miniconda3/envs/g1fetch`, Python 3.10.21.
- Preserved: Jetson torch 2.11.0, torchvision 0.26.0, CUDA 12.6, cuDSS 0.8.0.10,
  system TensorRT 10.3.0, NumPy 1.26.4, Pinocchio 3.1.0, CasADi 3.6.7,
  CycloneDDS 0.10.2, and RealSense 2.58.4.10922.
- Power mode remained 15 W, four CPU cores online. No services or power settings changed.
- Model setup added 23 missing distributions; before/after inventories confirm **zero existing
  distribution version changes**. Client setup needed no installations. Subsequent model
  `--check` passed without invoking pip installation.

The complete bundle is in `playground/Pretrained_models/UnifoLM-WLA-1.0-Base`.
The [pinned Hugging Face revision](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base/tree/dedc0612c4a921446db3a8cb0864bb7f7433329e)
is `dedc0612c4a921446db3a8cb0864bb7f7433329e`. All file lengths and both LFS SHA-256
hashes matched official metadata. The checkpoint is 12,458,898,964 bytes, SHA-256:

```text
65ac8911ce5556631cb538fb4d3499d9131d95b1da4eb1ce970c4215997d3299
```

Two `pip check` findings existed before installation and remained identical afterward:
cuDSS declares a missing `cuda-toolkit` distribution; `dex-retargeting` declares torch 2.3.0.
The existing system CUDA/Jetson torch arrangement passes actual BF16 GPU arithmetic and robotics
imports. These metadata findings were recorded without replacing the working GPU stack.

## Measured behavior

The checkpoint loaded all 1,256 tensors / 6,229,360,182 parameters. Model construction and weight
loading took 27.33 seconds in the first full run. All four direct predictions and all twelve
HTTP predictions produced finite arrays of shape `[30, 54]` with WBT normalization.

| Synthetic input | Tokens | Warm inference median | Loopback round-trip median | Round-trip p95 |
|---|---:|---:|---:|---:|
| Head only | 192 | 1.635 s | 1.640 s | 1.796 s |
| Three views | 490 | 3.493 s | 3.503 s | 3.537 s |

Each API case used one warm-up followed by five measured requests, at 336×448 pixels per view.
The percentiles describe these five samples, not a long-duration performance guarantee.
The first direct head-only prediction took 5.003 seconds. Measurements exclude live camera,
DDS, kinematics, and network travel beyond G1's loopback interface.

Peak PyTorch allocated memory was 11.84 GiB. During the direct smoke test, `tegrastats` reported
peak RAM 15,012 / 15,656 MB, peak swap 499 MB, and maximum junction temperature 49.375 °C.
With the warmed server resident after the API test, about 512 MiB remained available;
additional processes and live perception need a separate memory assessment.

The 30-action chunk at 30 Hz covers one second. Both measured configurations exceed that budget;
the bounded live executor would reject expired predictions. Three-view VLM encoding alone takes
about two seconds, so accelerating only the action head cannot meet that budget at this setting.
TensorRT 10.3 imports successfully; no full-policy TensorRT engine was built or claimed.
Offboard inference or validated VLM optimization remains a next step for reducing latency.

Offline tests on the robot: **52 passed, 2 skipped**. Optional ONNX export / PEFT training
packages were not installed solely to expand the test suite.

## Server operations

At handoff, the manually launched server listens on **G1's** `127.0.0.1:8601`.
It accepts head-only observations with `--allow-missing-cameras`: an explicit camera ablation,
not validation of the original three-view policy. It has no robot command publisher.
It is not registered as a service and will not restart after reboot.

Run on G1:

```bash
cd /home/unitree/unifolm-wla
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate g1fetch
python - <<'PY'
from urllib.request import urlopen
from model_server.tools.msgpack_numpy import unpackb
with urlopen('http://127.0.0.1:8601/health', timeout=5) as response:
    print(unpackb(response.read()))
PY
tail -n 20 results/deployment-home/server.log
```

The PID is recorded in `results/deployment-home/server.pid`. Verify it still belongs to
`python -m deploy.g1.server` before stopping it; PID files can become stale:

```bash
ps -p "$(cat results/deployment-home/server.pid)" -o pid,args
kill "$(cat results/deployment-home/server.pid)"
```

After stopping that process, restart in the foreground with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
python -u -m deploy.g1.server \
  --checkpoint playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --host 127.0.0.1 --port 8601 --allow-missing-cameras
```

Do not start a second full-model process alongside the resident server: this machine cannot
hold two copies. After stopping it, saved synthetic observations can be replayed with the
existing offline benchmark command from the README.

Raw logs, inventories, checksums, synthetic inputs/predictions, and the repeatable smoke script
are under `results/deployment-home/` on G1. An archive was copied to the same workspace-relative
directory on the office server. The compact [measurement record](reports/2026-10-05-onboard.json)
is tracked with this report.
