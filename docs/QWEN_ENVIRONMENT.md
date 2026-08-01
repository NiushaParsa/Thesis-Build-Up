# Qwen experiment environment

## Environment decision

The original `.venv` is a protected Python 3.9.12 environment used to
reproduce the Logistic Regression, MLP, retrieval-F1 Oracle, and retrieval
experiments. It was not rebuilt, renamed, deleted, or upgraded.

Interpreter discovery found Python 3.10.7 at
`C:\Users\behno\AppData\Local\Programs\Python\Python310\python.exe` and
Python 3.12.6 at
`C:\Users\behno\AppData\Local\Programs\Python\Python312\python.exe`.
Python 3.11 was not installed. Following the requested fallback rule,
`.venv-qwen` was created with Python 3.10.7. It is reserved for pretrained
Qwen Phase 1 and the later fine-tuned Qwen Phase 2; it is not used to rerun
the earlier baselines.

The Qwen executable is
`C:\Users\behno\Repos\Thesis Build Up\.venv-qwen\Scripts\python.exe`.
The system Python environment was not modified.

## Reproducible creation

```powershell
C:\Users\behno\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv-qwen
.\.venv-qwen\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-qwen\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-qwen\Scripts\python.exe -m pip install -r requirements-qwen.txt
```

`requirements-qwen.txt` is the minimal direct dependency manifest.
`reports/qwen_pretrained_zero_shot_router_evidence_length_oracle/qwen_environment_lock.txt`
records every resolved package. The pre-existing Python 3.9 package inventory
is preserved separately in `environment_before_transformers_upgrade.txt`.

Primary versions retained from the legacy environment where compatible are
PyTorch 2.8.0 CPU, NumPy 2.0.2, Qdrant client 1.16.1, OpenAI 2.19.0,
python-dotenv 1.2.1, and tqdm 4.67.3. Transformers necessarily differs:
the legacy environment has 4.57.6, while Qwen uses `5.15.0.dev0` from commit
`2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`. Its resolver also selected
newer supporting packages including Hugging Face Hub 1.26.0, grpcio 1.83.0,
protobuf 7.35.1, pydantic 2.13.4, regex 2026.7.19, pywin32 312, and fsspec
2026.4.0. Exact transitive versions are authoritative in the lock file.

## Verified checks

Dependency imports, read-only Qdrant connectivity, processor/model loading,
and deterministic one-prompt generation all passed. The existing Qdrant
service at `127.0.0.1:6334` and its collections are reused without schema or
index changes. No collection is created, deleted, rebuilt, re-indexed, or
repointed.

The exact post-trained model is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. The Base model is not used.
Model snapshots live under the git-ignored
`tmp/huggingface_qwen_cache` directory. Inference is CPU-only,
`torch.bfloat16`, unquantized, deterministic, frozen, and performed under
`torch.inference_mode()`.

Before the full run, missing package metadata in the workspace-restored
`.venv-qwen` was repaired in place from the pinned CPU wheels and
`requirements-qwen.txt`; the environment was not rebuilt. The resolver moved
Hugging Face Hub from the smoke-run transitive version 1.25.1 to compatible
1.26.0. The exact model revision, Transformers commit, tokenizer, prompt,
parser, dtype, device, and decoding configuration did not change.

## Required records

- Direct dependencies: `requirements-qwen.txt`
- Exact resolved packages:
  `reports/qwen_pretrained_zero_shot_router_evidence_length_oracle/qwen_environment_lock.txt`
- Legacy snapshot:
  `reports/qwen_pretrained_zero_shot_router_evidence_length_oracle/environment_before_transformers_upgrade.txt`
- Machine/run environment:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/configuration/environment.json`
- Fixed prompt:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/configuration/fixed_prompt.json`
- Smoke configuration and timings:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/smoke/summary.json`
- Experiment report:
  `reports/qwen_pretrained_zero_shot_router_evidence_length_oracle/experiment_report.md`
