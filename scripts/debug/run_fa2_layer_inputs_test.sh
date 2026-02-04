#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "Deep FA2 diagnostic: capture exact inputs/outputs at each layer." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --no-host-networking \
    --gpus 1 -- uv run python scripts/debug/test_fa2_layer_inputs.py
