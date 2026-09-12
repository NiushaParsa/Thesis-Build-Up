#!/usr/bin/env python
"""Run one frozen Phase 5B-v2 Qwen split at a time.

This staging helper lets the training-output validity gate be checked before
the validation questions are processed.  It does not change the frozen model,
prompt, decision rule, inputs, or downstream experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import similarity_tree_phase5b_v2 as phase5b_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "validation"))
    args = parser.parse_args()

    phase5b_v2.configure_base()
    base = phase5b_v2.base
    audit = base.preflight(phase5b_v2.OUTPUT_ROOT, require_qdrant=False)
    lock_path = phase5b_v2.OUTPUT_ROOT / "configuration" / "procedure_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["source_hashes"] != audit["source_hashes"]:
            raise RuntimeError("Frozen Phase 5B-v2 source hashes changed")
    else:
        base.write_procedure_lock(phase5b_v2.OUTPUT_ROOT, audit)

    result = base.run_qwen_inference(phase5b_v2.OUTPUT_ROOT, (args.split,))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
