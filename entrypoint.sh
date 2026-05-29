#!/bin/bash
set -euo pipefail

MODEL_FILE="${MODEL_FILE:-/models/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q8_0.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-/models/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-model-f16.gguf}"
LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-16384}"
N_THREADS="${N_THREADS:-4}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

cleanup() {
    if [[ -n "${LLAMA_PID:-}" ]]; then
        kill "$LLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

/app/llama-server \
    --model "$MODEL_FILE" \
    --mmproj "$MMPROJ_FILE" \
    --host "$LLAMA_HOST" \
    --port "$LLAMA_PORT" \
    --ctx-size "$CTX_SIZE" \
    --threads "$N_THREADS" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --no-jinja &
LLAMA_PID=$!

READY=0
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" > /dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "llama-server exited unexpectedly" >&2
        exit 1
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    echo "llama-server failed to become ready" >&2
    exit 1
fi

cd /service
exec uvicorn main:app --host 0.0.0.0 --port 8000
