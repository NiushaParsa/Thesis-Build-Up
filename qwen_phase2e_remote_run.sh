#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

PROJECT=/workspace/thesis-granularity-router
CODE=/dev/shm/phase2e_code
STUDY=/dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle
PYTHON=${PROJECT}/.venv-qwen/bin/python
ENTRY=${CODE}/qwen_phase2e_sequence_classifier_lr_grid.py

exec 9>/dev/shm/qwen_phase2e_lr_grid.lock
if ! flock -n 9; then
  echo "Another Phase 2E grid process already holds the study lock" >&2
  exit 75
fi

cat /etc/vast-agents-guide.md >/dev/null
test -x "${PYTHON}"
test -f "${ENTRY}"
test -f "${CODE}/phase2e_source_manifest.sha256"
(cd "${CODE}" && sha256sum --check phase2e_source_manifest.sha256)

shm_available_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
workspace_available_kib=$(df -Pk /workspace | awk 'NR==2 {print $4}')
if (( shm_available_kib < 33554432 )); then
  echo "Phase 2E requires at least 32 GiB free in /dev/shm" >&2
  exit 70
fi
if (( workspace_available_kib < 3145728 )); then
  echo "Phase 2E requires at least 3 GiB free in /workspace" >&2
  exit 70
fi

cd "${PROJECT}"
export HF_HOME=/dev/shm/qwen_phase2e_hf
export MPLCONFIGDIR=/dev/shm/phase2e_mpl
export PYTHONPATH="${CODE}"
export PHASE2E_REPOSITORY_BASE_COMMIT=12c7b1a22f552f83d54a752f87f6687c98b52944
export PHASE2E_REMOTE_RUNNER_SHA256
PHASE2E_REMOTE_RUNNER_SHA256=$(sha256sum "$0" | awk '{print $1}')

"${PYTHON}" - <<'PY'
import importlib.metadata as metadata
import json
import pathlib
import sys

import torch
import transformers

assert sys.version_info[:3] == (3, 10, 7), sys.version
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.cuda.get_device_name(0) == "NVIDIA A100-SXM4-40GB"
assert transformers.__version__ == "5.15.0.dev0", transformers.__version__
distribution = metadata.distribution("transformers")
direct_url = next(
    pathlib.Path(distribution.locate_file(item))
    for item in distribution.files or []
    if str(item).endswith("direct_url.json")
)
source = json.loads(direct_url.read_text())
assert source["vcs_info"]["commit_id"] == (
    "2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7"
)
print("Phase 2E environment gate passed")
PY

phase2e() {
  "${PYTHON}" -u "${ENTRY}" --study-root "${STUDY}" "$@"
}

phase2e prepare
for variant in lr5e-6 lr1e-5 lr2e-5; do
  phase2e inspect --variant "${variant}"
done

for variant in lr5e-6 lr1e-5 lr2e-5; do
  case "${variant}" in
    lr5e-6)
      run_id=qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1
      ;;
    lr1e-5)
      run_id=qwen-phase2e-base-sequence-classifier-token-count-prompt-lr1e-5-5epochs-full-parameter-20260808-seed42-v1
      ;;
    lr2e-5)
      run_id=qwen-phase2e-base-sequence-classifier-token-count-prompt-lr2e-5-5epochs-full-parameter-20260808-seed42-v1
      ;;
    *)
      echo "Unsupported Phase 2E variant: ${variant}" >&2
      exit 2
      ;;
  esac

  trial_root=${STUDY}/trials/${variant}
  run_dir=${trial_root}/runs/${run_id}
  summary=${run_dir}/summary.json
  if [[ -f "${summary}" ]]; then
    phase2e audit-completed --variant "${variant}"
    continue
  fi

  if [[ -d "${run_dir}" ]]; then
    if [[ -f "${run_dir}/checkpoint_manifest.json" ]]; then
      phase2e resume-latest --variant "${variant}"
      continue
    fi
    recovery=${trial_root}/recovery
    mkdir -p "${recovery}"
    moved=${recovery}/${run_id}-no-checkpoint-$(date -u +%Y%m%dT%H%M%SZ)
    test ! -e "${moved}"
    mv "${run_dir}" "${moved}"
    echo "Moved checkpoint-free interrupted run to ${moved}"
  fi

  phase2e train --variant "${variant}"
done

phase2e select
if [[ ! -f "${STUDY}/comparison/selected_final_summary.json" ]]; then
  phase2e final-selected
fi

echo "Phase 2E classification grid and locked-winner final validation completed."
