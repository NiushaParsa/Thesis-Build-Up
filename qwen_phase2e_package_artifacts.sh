#!/bin/bash
set -euo pipefail

STUDY=/dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle
BASE=${STUDY##*/}
XFER=/dev/shm/phase2e_transfer_v1
PY=/workspace/thesis-granularity-router/.venv-qwen/bin/python

cat /etc/vast-agents-guide.md >/dev/null
supervisor_rc=0
status_line=$(supervisorctl status qwen-phase2e-lr-grid 2>&1) || supervisor_rc=$?
if (( supervisor_rc != 0 && supervisor_rc != 3 )); then
  echo "Could not query Phase 2E runner status: ${status_line}" >&2
  exit 1
fi
status=$(awk '{print $2}' <<<"${status_line}")
if [[ "${status}" != EXITED && "${status}" != STOPPED ]]; then
  echo "Phase 2E runner is still ${status}" >&2
  exit 1
fi
test -f "${STUDY}/comparison/selected_trial.json"
test -f "${STUDY}/comparison/selected_final_summary.json"
test ! -e "${XFER}"

available_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
if (( available_kib < 25165824 )); then
  echo "Artifact packaging requires at least 24 GiB free in /dev/shm" >&2
  exit 70
fi
mkdir -p "${XFER}/manifests" "${XFER}/archives" "${XFER}/metadata_chunks"

"${PY}" - "${STUDY}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
runs = {
    "lr5e-6": "qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1",
    "lr1e-5": "qwen-phase2e-base-sequence-classifier-token-count-prompt-lr1e-5-5epochs-full-parameter-20260808-seed42-v1",
    "lr2e-5": "qwen-phase2e-base-sequence-classifier-token-count-prompt-lr2e-5-5epochs-full-parameter-20260808-seed42-v1",
}
for variant, run_id in runs.items():
    run = root / "trials" / variant / "runs" / run_id
    summary = json.loads((run / "summary.json").read_text())
    best = json.loads((run / "best_checkpoint.json").read_text())
    assert summary["status"] == "complete"
    assert summary["selected_checkpoint_id"] == best["checkpoint_id"]
    checkpoints = sorted(
        path.name for path in (run / "checkpoints").iterdir() if path.is_dir()
    )
    assert checkpoints == [best["checkpoint_id"]]
    assert (
        run / "checkpoints" / best["checkpoint_id"] / "model" / "model.safetensors"
    ).is_file()
final = json.loads(
    (root / "comparison" / "selected_final_summary.json").read_text()
)
assert final["status"] == "selected_checkpoint_final_validation_complete"
print("Phase 2E completion and checkpoint-retention gate passed")
PY

printf 'variant\trun_id\tcheckpoint_id\trelative_path\tarchive_bytes\tarchive_sha256\tchunk_count\n' \
  > "${XFER}/manifests/transfer_inventory.tsv"
declare -A RUN=(
  [lr5e-6]=qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1
  [lr1e-5]=qwen-phase2e-base-sequence-classifier-token-count-prompt-lr1e-5-5epochs-full-parameter-20260808-seed42-v1
  [lr2e-5]=qwen-phase2e-base-sequence-classifier-token-count-prompt-lr2e-5-5epochs-full-parameter-20260808-seed42-v1
)
for variant in lr5e-6 lr1e-5 lr2e-5; do
  run_id=${RUN[${variant}]}
  best=${STUDY}/trials/${variant}/runs/${run_id}/best_checkpoint.json
  checkpoint_id=$("${PY}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_id"])' \
    "${best}")
  relative=${BASE}/trials/${variant}/runs/${run_id}/checkpoints/${checkpoint_id}
  archive_relative=archives/phase2e-${variant}-${checkpoint_id}.tar.zst
  archive=${XFER}/${archive_relative}
  chunks=${XFER}/${variant}_chunks
  mkdir "${chunks}"
  (
    cd /dev/shm
    find "${relative}" -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "${XFER}/manifests/${variant}_selected_checkpoint_files.sha256"
  tar --zstd -cf "${archive}" -C /dev/shm "${relative}"
  (
    cd "${XFER}"
    sha256sum "${archive_relative}"
  ) > "${XFER}/manifests/${variant}_archive.sha256"
  split -b 100M -d -a 3 "${archive}" \
    "${chunks}/phase2e-${variant}-${checkpoint_id}.tar.zst.part-"
  (
    cd "${XFER}"
    find "${variant}_chunks" -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "${XFER}/manifests/${variant}_chunks.sha256"
  bytes=$(stat -c%s "${archive}")
  digest=$(sha256sum "${archive}" | awk '{print $1}')
  count=$(find "${chunks}" -type f | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${run_id}" "${checkpoint_id}" "${relative}" \
    "${bytes}" "${digest}" "${count}" \
    >> "${XFER}/manifests/transfer_inventory.tsv"
done

(
  cd /dev/shm
  find "${BASE}" -type f \
    ! -path "${BASE}/trials/*/runs/*/checkpoints/*" \
    ! -path "${BASE}/trials/*/tensorboard/*" \
    -print0 | sort -z | xargs -0 sha256sum
) > "${XFER}/manifests/metadata_files.sha256"
tar --zstd -cf "${XFER}/archives/phase2e-metadata.tar.zst" -C /dev/shm \
  --exclude="${BASE}/trials/*/runs/*/checkpoints" \
  --exclude="${BASE}/trials/*/runs/*/checkpoints/*" \
  --exclude="${BASE}/trials/*/tensorboard" \
  --exclude="${BASE}/trials/*/tensorboard/*" \
  "${BASE}"
(
  cd "${XFER}"
  sha256sum archives/phase2e-metadata.tar.zst
) > "${XFER}/manifests/metadata_archive.sha256"
split -b 100M -d -a 3 "${XFER}/archives/phase2e-metadata.tar.zst" \
  "${XFER}/metadata_chunks/phase2e-metadata.tar.zst.part-"
(
  cd "${XFER}"
  find metadata_chunks -type f -print0 | sort -z | xargs -0 sha256sum
) > "${XFER}/manifests/metadata_chunks.sha256"
(
  cd "${XFER}"
  find manifests -type f -print0 | sort -z | xargs -0 sha256sum
) > "${XFER}/manifest_bundle.sha256"

du -sh "${STUDY}" "${XFER}"
cat "${XFER}/manifests/transfer_inventory.tsv"
sha256sum "${XFER}/archives/phase2e-metadata.tar.zst"
