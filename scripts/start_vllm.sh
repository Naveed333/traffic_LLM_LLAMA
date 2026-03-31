#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server with specified model and context window.
# Usage: ./start_vllm.sh <MODEL_ID> <MAX_MODEL_LEN>
# Example: ./start_vllm.sh meta-llama/Llama-3.1-8B-Instruct 131072

set -euo pipefail

MODEL_ID="${1:?Usage: $0 <MODEL_ID> <MAX_MODEL_LEN>}"
MAX_MODEL_LEN="${2:?Usage: $0 <MODEL_ID> <MAX_MODEL_LEN>}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
HEALTH_URL="http://localhost:${PORT}/v1/models"
MAX_WAIT_SECONDS=600

echo "============================================================"
echo "Starting vLLM server"
echo "  Model:          ${MODEL_ID}"
echo "  Max model len:  ${MAX_MODEL_LEN}"
echo "  Host:port:      ${HOST}:${PORT}"
echo "============================================================"

# Kill any existing vLLM process on the port
if lsof -ti tcp:"${PORT}" &>/dev/null; then
    echo "Killing existing process on port ${PORT}..."
    lsof -ti tcp:"${PORT}" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start vLLM server in the background
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --enforce-eager \
    &

VLLM_PID=$!
echo "vLLM server started with PID ${VLLM_PID}"

# Wait for server to become healthy
echo "Waiting for vLLM server to be ready (max ${MAX_WAIT_SECONDS}s)..."
elapsed=0
while [ "${elapsed}" -lt "${MAX_WAIT_SECONDS}" ]; do
    if curl -sf "${HEALTH_URL}" &>/dev/null; then
        echo ""
        echo "✓ vLLM server is ready at http://localhost:${PORT}/v1"
        echo "  Model: ${MODEL_ID}, Context: ${MAX_MODEL_LEN}"
        exit 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo -n "."
done

echo ""
echo "✗ vLLM server did not become ready within ${MAX_WAIT_SECONDS}s. Check logs."
kill "${VLLM_PID}" 2>/dev/null || true
exit 1
