#!/bin/bash
set -eo pipefail

source configs/beaker_configs/ray_node_setup.sh

uv run python scripts/debug/minimal_doc_lens_repro.py
