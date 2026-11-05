#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Convenience script to launch the Qwen3.8 Flash DGX UltraFast server on NVIDIA GB10 / DGX Spark.
# Recipe: https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast
set -euo pipefail

RECIPE_DIR="${HOME}/qwen3.8-Flash-DGX-UltraFast"

if [ ! -d "$RECIPE_DIR" ]; then
  echo "Error: Recipe directory not found at $RECIPE_DIR" >&2
  exit 1
fi

echo "Launching Qwen3.8 Flash DGX UltraFast on port 8000..."
cd "$RECIPE_DIR"

# Launch using the recipe's promoted v16b configuration
bash recipe/config/v16b/launch.sh --run "$@"

