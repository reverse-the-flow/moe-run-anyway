#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_SERVER_BIN:=llama-server}"
: "${LLAMA_SERVER_INTERNAL_HOST:=127.0.0.1}"
: "${LLAMA_SERVER_INTERNAL_PORT:=18080}"
: "${SIDECAR_LISTEN_HOST:=0.0.0.0}"
: "${SIDECAR_LISTEN_PORT:=8080}"
: "${SIDECAR_RUNS_DIR:=/var/log/memory-moe-sidecar}"
: "${SIDECAR_LABEL:=llama-sidecar}"
: "${SIDECAR_CAPTURE_GPU:=1}"
: "${SIDECAR_EXTRA_ARGS:=}"

mkdir -p "${SIDECAR_RUNS_DIR}"

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${SIDECAR_PID:-}" ]] && kill -0 "${SIDECAR_PID}" 2>/dev/null; then
    kill "${SIDECAR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${LLAMA_PID:-}" ]] && kill -0 "${LLAMA_PID}" 2>/dev/null; then
    kill "${LLAMA_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

terminate() {
  cleanup
  exit 0
}

trap cleanup EXIT
trap terminate INT TERM

"${LLAMA_SERVER_BIN}" \
  --host "${LLAMA_SERVER_INTERNAL_HOST}" \
  --port "${LLAMA_SERVER_INTERNAL_PORT}" \
  "$@" &
LLAMA_PID=$!

sidecar_cmd=(
  python3
  /opt/memory-moe/llama_sidecar.py
  --listen-host "${SIDECAR_LISTEN_HOST}"
  --listen-port "${SIDECAR_LISTEN_PORT}"
  --upstream-base-url "http://${LLAMA_SERVER_INTERNAL_HOST}:${LLAMA_SERVER_INTERNAL_PORT}"
  --output-dir "${SIDECAR_RUNS_DIR}"
  --label "${SIDECAR_LABEL}"
)

if [[ "${SIDECAR_CAPTURE_GPU}" == "1" ]]; then
  sidecar_cmd+=(--capture-gpu)
fi

if [[ -n "${SIDECAR_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${SIDECAR_EXTRA_ARGS} )
  sidecar_cmd+=("${extra_args[@]}")
fi

"${sidecar_cmd[@]}" &
SIDECAR_PID=$!

while true; do
  if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
    wait "${LLAMA_PID}"
  fi
  if ! kill -0 "${SIDECAR_PID}" 2>/dev/null; then
    wait "${SIDECAR_PID}"
  fi
  sleep 1
done
