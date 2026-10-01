#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [TEST_ARGS...]" >&2
    exit 1
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

PYTHONPATH="$(dirname "$0")/..${PYTHONPATH:+:$PYTHONPATH}" \
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$(dirname "$0")/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "$@"
