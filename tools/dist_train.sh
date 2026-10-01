#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 CONFIG GPUS [TRAIN_ARGS...]" >&2
    exit 1
fi

CONFIG=$1
GPUS=$2
shift 2

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
    "$(dirname "$0")/train.py" \
    "$CONFIG" \
    --launcher pytorch \
    "$@"
