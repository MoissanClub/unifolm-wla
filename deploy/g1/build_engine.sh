#!/usr/bin/env bash
# Run on the destination Jetson. TensorRT 10.3, ONNX plus external weight files.
set -euo pipefail
onnx_path=${1:?Usage: build_engine.sh action.onnx action.engine}
engine_path=${2:?Supply the destination engine path}
trtexec_bin=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
workspace_args=()
if [[ -n ${TRT_WORKSPACE_MIB:-} ]]; then
  workspace_args+=(--memPoolSize="workspace:$TRT_WORKSPACE_MIB")
fi
"$trtexec_bin" --onnx="$onnx_path" --saveEngine="$engine_path" --fp16 \
  "${workspace_args[@]}" --skipInference
