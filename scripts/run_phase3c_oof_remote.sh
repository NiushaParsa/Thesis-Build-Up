#!/usr/bin/env bash
set -euo pipefail

repository="${1:-/workspace/thesis-granularity-router}"
cd "$repository"
source .venv-fusion/bin/activate

python qwen_phase3c_oof.py audit
for fold in 0 1 2 3 4; do
  python qwen_phase3c_oof.py train-fold --fold "$fold"
done
python qwen_phase3c_oof.py assemble-oof
python qwen_phase3c_oof.py train-full
python qwen_phase3c_oof.py freeze
python qwen_phase3c_oof.py extract-validation
python qwen_phase3c_oof.py train-evaluate
