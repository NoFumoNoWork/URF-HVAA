#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m src.minimal_demo --output_dir outputs/minimal_demo
