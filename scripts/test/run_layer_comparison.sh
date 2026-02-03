#!/bin/bash
set -eo pipefail

source configs/beaker_configs/ray_node_setup.sh

uv run python scripts/debug/compare_models_layer_by_layer.py
