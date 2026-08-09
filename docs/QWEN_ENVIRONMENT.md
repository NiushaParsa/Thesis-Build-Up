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
Qwen Phase 1 and the completed fine-tuned Qwen Phase 2, Phase 2B, Phase 2C,
and Phase 2D experiments, as well as the Phase 2E classification grid; it is
not used to rerun the earlier baselines.

The Qwen executable is
`C:\Users\behno\Repos\Thesis Build Up\.venv-qwen\Scripts\python.exe`.
The system Python environment was not modified.

## Phase 1 local CPU environment

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

### Verified Phase 1 checks

Dependency imports, read-only Qdrant connectivity, processor/model loading,
and deterministic one-prompt generation all passed. The existing Qdrant
service at `127.0.0.1:6334` and its collections are reused without schema or
index changes. No collection is created, deleted, rebuilt, re-indexed, or
repointed.

The exact pretrained model is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. The Base model is not used.
Model snapshots live under the git-ignored
`tmp/huggingface_qwen_cache` directory. Phase 1 inference is CPU-only,
`torch.bfloat16`, unquantized, deterministic, frozen, and performed under
`torch.inference_mode()`.

Before the full run, missing package metadata in the workspace-restored
`.venv-qwen` was repaired in place from the pinned CPU wheels and
`requirements-qwen.txt`; the environment was not rebuilt. The resolver moved
Hugging Face Hub from the smoke-run transitive version 1.25.1 to compatible
1.26.0. The exact model revision, Transformers commit, tokenizer, prompt,
parser, dtype, device, and decoding configuration did not change.

## Phase 2 remote CUDA environment

Phase 2 used a separate `.venv-qwen` inside the remote project at
`/workspace/thesis-granularity-router/.venv-qwen`. Its executable was
`/workspace/thesis-granularity-router/.venv-qwen/bin/python`, and its exact
Python version was
`3.10.7 (main, Oct  3 2022, 02:19:58) [Clang 14.0.3 ]`.
This environment did not replace or modify either the protected local legacy
`.venv` or the local Phase 1 `.venv-qwen`.

The remote run used:

- `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`;
- Transformers `5.15.0.dev0`, installed from commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`;
- PyTorch `2.8.0+cu128`, torchvision `0.23.0+cu128`, and CUDA 12.8;
- TensorBoard `2.20.0`;
- one `NVIDIA A100-SXM4-40GB` on device `cuda`;
- `torch.bfloat16` with no quantization.

`requirements-qwen-phase2.txt` extends the minimal Qwen dependency manifest
with TensorBoard only. The exact resolved remote inventory is frozen in
`outputs/qwen_finetuned_router_evidence_length_oracle/environment/phase2_package_lock_after.txt`.
An equivalent clean remote environment is created without touching a system
Python installation by:

```bash
uv venv --python 3.10.7 .venv-qwen
uv pip install --python .venv-qwen/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-qwen/bin/python -r requirements-qwen-phase2.txt
```

Phase 2 performed full-parameter supervised fine-tuning. All 852,985,920
parameters were trainable; no LoRA, QLoRA, adapters, prompt tuning, separate
classification head, or quantization was used. The model received only the
same fixed routing instruction and original question text. Evidence, evidence
length, answers, paper text, embeddings, retrieved chunks, scores, metadata,
and handcrafted features were absent. Training loss was restricted to the
assistant target tokens. The run executed 213 optimizer steps across three
epochs with deterministic CUDA settings. Generation used the frozen parser,
greedy `do_sample=false` decoding, and `max_new_tokens=8`.

The run configuration was batch size 4, gradient accumulation 8 (effective
batch size 32), maximum sequence length 128, learning rate 2e-5, weight decay
0.01, cosine scheduling, 5% warmup, gradient clipping at 1.0, and seed 42.
Strict CUDA determinism used `CUBLAS_WORKSPACE_CONFIG=:4096:8` and PyTorch
deterministic algorithms.
Full validation and checkpointing occurred at each epoch. The selected
checkpoint was `step-000213`, chosen by validation macro-F1. Its model tensor
SHA-256 is
`7d23db1fde0c621623a7d4030073e8858854eba9a4b2d3d7bccda8ca730e2c45`;
optimizer, scheduler, and random-state files are also present. Five recorded
checkpoint generation probes repeated exactly. The selected checkpoint was
copied back and verified locally: 11 files, 4,735,895,186 bytes (4.411 GiB),
with every SHA-256 matching `selected_checkpoint_sha256.txt`.

Training, including three validation passes and checkpoint writes, took
2,107.3131887838244 seconds. Peak allocated/reserved GPU memory was
10.660949230194092/11.943359375 GiB. Reloaded final validation reported
1.6835732460021973 GiB peak allocated GPU memory and
1.6948738098144531 GiB RSS. The TensorBoard audit reconciled 213 structured
training steps and three validation events with zero loss, scalar-value, or
scalar-count mismatches and independently reproduced the selected checkpoint.

Qdrant was not moved to the remote trainer and no remote collection was
created. After the selected-checkpoint predictions were copied back, the
unchanged local Qdrant service at `127.0.0.1:6334` was used for Phase 2
same-paper top-five retrieval. No collection, port, storage path, schema,
record, or index was created, deleted, rebuilt, or changed.

The recorded execution sequence is:

```bash
.venv-qwen/bin/python qwen_phase2.py inspect-data
.venv-qwen/bin/python qwen_phase2.py train --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/python qwen_phase2.py audit-tensorboard --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/python qwen_phase2.py verify-checkpoint --run-id qwen-phase2-full-parameter-20260802-seed42-v2 --checkpoint outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoints/step-000213
.venv-qwen/bin/python qwen_phase2.py final-validation --run-id qwen-phase2-full-parameter-20260802-seed42-v2
```

The local retrieval and final integrity audit are reproduced with:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2.py evaluate-retrieval --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.\.venv-qwen\Scripts\python.exe qwen_phase2.py audit-final --run-id qwen-phase2-full-parameter-20260802-seed42-v2
```

## Phase 2B remote CUDA environment

Phase 2B reused the isolated remote `.venv-qwen` described above; it did not
modify the protected local legacy `.venv`, the local Phase 1 environment, the
system Python installation, or the pinned Phase 2 dependency set. Both Phase
2B variants record Python 3.10.7 at
`/workspace/thesis-granularity-router/.venv-qwen/bin/python`, Transformers
`5.15.0.dev0` at commit
`2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`, PyTorch `2.8.0+cu128`, CUDA
12.8, TensorBoard 2.20.0, and one `NVIDIA A100-SXM4-40GB`. Training and final
validation used CUDA with `torch.bfloat16`, no quantization, seed 42, and
strict deterministic settings.

The model remains `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`; all 852,985,920 parameters were
trainable. The Phase 2B formulation uses five verified single-token aliases
and deterministic restricted-logit classification, so it does not depend on a
new runtime package, unrestricted generation, or the legacy parser. Both
variants used maximum sequence length 128, per-device batch 4, gradient
accumulation 8, effective batch 32, AdamW at `2e-5`, weight decay 0.01,
cosine scheduling with 5% warmup, gradient clipping 1.0, three epochs, and 213
optimizer updates. Phase 2B-B's effective-number class weighting changes the
loss computation only; it does not change the environment.

The two immutable run identities are:

- `qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1`, under
  `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/`;
- `qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1`, under
  `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/`.

Remote inspection, training, and final validation are reproduced with:

```bash
.venv-qwen/bin/python qwen_phase2b.py inspect --variant alias-unweighted
.venv-qwen/bin/python qwen_phase2b.py inspect --variant alias-classbalanced
.venv-qwen/bin/python qwen_phase2b.py train --variant alias-unweighted --mode full --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py train --variant alias-classbalanced --mode full --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py final-validation --variant alias-unweighted --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py final-validation --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
```

As in Phase 2, Qdrant was not installed or moved onto the trainer. Canonical
predictions were evaluated later against the unchanged local Qdrant service
at `127.0.0.1:6334`; no collection, schema, record, index, port, or storage
path was changed. The local commands are:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py evaluate-retrieval --variant alias-unweighted --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py evaluate-retrieval --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py compare --output outputs\qwen_phase2b_comparison_evidence_length_oracle\four_way_comparison.json
```

The saved retrieval wall times are 178.27286399999866 seconds for Phase 2B-A
and 377.0999227000284 seconds for Phase 2B-B. The Phase 2B-B local retrieval
overlapped transfer of its selected-checkpoint archive. These exact values are
retained for provenance, but their difference is not a model- or
environment-speed comparison. Saved training and final-validation timings are
isolated measurements.

Manual post-transfer QA used `rsync` checksum dry-runs over every copied
training, classification, validation, TensorBoard, and checkpoint file,
excluding the locally retrieval-extended `final_summary.json`. Phase 2B-A had
no differences. Phase 2B-B had two stale local preflight copies
(`configuration/experiment.json` and `configuration/preflight_manifest.json`)
whose semantic contents were unchanged and only generated timestamps differed;
they were replaced from the GPU source, and the targeted checksum rerun found
no differences. The selected A/B checkpoints contain 11 files totaling
4,735,895,574/4,735,895,530 bytes. No standalone Phase 2B hash-inventory file
was saved.

Completed-summary replay revalidated all 924 retrieval records in each run.
Qdrant counts were unchanged before and after: `PaperChunk` 1,701,822,
`PaperEvidence` 9,522, `PaperQuestion` 4,526, `RouterDataset` 3,170,
`RetrievalEvaluation` 18,622, `Stage4VerifyRetrievalEval` 10,
`Stage4VerifyRouterDataset` 2, and `Stage5MixedEvaluation` 2.

## Phase 2C remote CUDA environment

Phase 2C used the same isolated remote executable,
`/workspace/thesis-granularity-router/.venv-qwen/bin/python`, with exact Python
version `3.10.7 (main, Oct  3 2022, 02:19:58) [Clang 14.0.3 ]`. Its recorded
software and hardware environment is:

- Transformers `5.15.0.dev0`, commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`;
- PyTorch `2.8.0+cu128`, CUDA 12.8, TensorBoard `2.20.0`;
- one `NVIDIA A100-SXM4-40GB` on `cuda`;
- `torch.bfloat16`, no quantization;
- strict deterministic execution with seed 42.

The model identity deliberately changes to `Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. It is loaded with
`AutoModelForSequenceClassification`, `num_labels=5`, a bias-free 5×1024
`score.weight` classifier head, right padding, and pad token ID 248044. The
five logits map directly to 10, 20, 40, 80, and 160. Phase 2C performs no
generation and uses no chat template or parser.

The only input is the fixed supervisor instruction plus the original question
text, formatted as `{instruction}\n\nQuestion: {original_question_text}`. Train
sequence lengths are 86--112 tokens; validation lengths are 87--115, so no
example exceeds the fixed maximum length of 128. No evidence, answer, paper
text, retrieved chunks, retrieval scores, embedding, metadata, or handcrafted
feature enters the model.

All 852,991,040 parameters were marked trainable. Uniform five-class
cross-entropy, AdamW, batch size 4, gradient accumulation 8, effective batch
32, learning rate 2e-5, weight decay 0.01, cosine scheduling, 5% warmup (11
steps), clipping 1.0, three fixed epochs, and 213 optimizer updates were used.
The gradient-coverage audit passed: the language backbone and classifier head
received gradients. Because this was a text-only path through a composite
model, the vision tower did not receive gradients; 752,398,144 parameters had
gradients and 100,592,896 did not. This distinction is recorded rather than
claiming that image-path parameters were updated.

Training elapsed time was 1276.56244828552 seconds, with peak allocated/reserved
GPU memory 8.96875286102295/9.517578125 GiB and recorded RSS
1.96734619140625 GiB. Selected-checkpoint reload took 2.5492455568164587
seconds and isolated validation inference took 33.99719780869782 seconds.
Final-validation peak allocated/reserved memory was
1.715855598449707/1.77734375 GiB with RSS 1.6998367309570312 GiB. Local
retrieval took 134.9306207000045 seconds; the known training, final-validation,
and retrieval duration is 1448.0395123510389 seconds.

The recorded CUDA sequence, using the RAM-backed training root, is:

```bash
.venv-qwen/bin/python qwen_phase2c_sequence_classifier.py --output-root /dev/shm/qwen_phase2c_sequence_classifier_evidence_length_oracle inspect
.venv-qwen/bin/python qwen_phase2c_sequence_classifier.py --output-root /dev/shm/qwen_phase2c_sequence_classifier_evidence_length_oracle train --mode full --run-id qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1
.venv-qwen/bin/python qwen_phase2c_sequence_classifier.py --output-root /dev/shm/qwen_phase2c_sequence_classifier_evidence_length_oracle final-validation --run-id qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1
```

Canonical predictions were copied back and evaluated through the unchanged
local Qdrant service at `127.0.0.1:6334`. No Qdrant collection, record, schema,
index, port, or storage path was created, deleted, rebuilt, or changed. Local
commands are:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2c_posttraining.py evaluate-retrieval --run-id qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2c_posttraining.py compare --output outputs\qwen_phase2c_comparison_evidence_length_oracle\five_way_comparison.json
```

The original protected local Python 3.9 `.venv`, the local Phase 1 CPU
environment, system Python, and all earlier experiment artifacts remained
separate and unchanged.

## Phase 2D remote CUDA environment

Phase 2D reused the Phase 2C remote CUDA environment without changing either
the legacy `.venv` or `.venv-qwen`. The recorded executable is
`/workspace/thesis-granularity-router/.venv-qwen/bin/python`, with exact Python
version `3.10.7 (main, Oct  3 2022, 02:19:58) [Clang 14.0.3 ]`. Software and
hardware provenance is:

- Transformers `5.15.0.dev0`, commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`;
- PyTorch `2.8.0+cu128`, CUDA 12.8, TensorBoard `2.20.0`;
- one `NVIDIA A100-SXM4-40GB` on `cuda`;
- `torch.bfloat16`, no quantization;
- strict deterministic execution with seed 42.

The model remains `Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, loaded with
`AutoModelForSequenceClassification`, five labels, a bias-free 5×1024
`score.weight` head, right padding, and pad token ID 248044. The fresh seed-42
head initialization hash is
`09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5`,
the same initial head identity recorded for Phase 2C. Class IDs 0--4 map to
10/20/40/80/160.

The only scientific change from Phase 2C is the instruction:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160 tokens. Return only the number

Its SHA-256 is
`b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5`.
The input template remains
`{instruction}\n\nQuestion: {original_question_text}` and the model receives
no evidence, evidence length, answer, paper text, retrieved chunk or score,
embedding, metadata, or handcrafted feature. No chat template, generation, or
parser is used. Train sequence lengths are 95--121 tokens and validation
lengths are 96--124, so all 3,169 examples remain within maximum length 128
without truncation.

The saved Phase 2C-to-Phase 2D protocol audit passed. It verifies the same
model/revision, architecture, seed, initial head, environment, uniform loss,
optimizer and schedule, checkpoint-selection rule, 2,245/924 preserved
examples, and frozen train/validation Oracle hashes. Only prompt identity and
necessary provenance fields differ; tokenized sequence lengths are an expected
consequence of the prompt change. The Phase 2D experiment fingerprint is
`dad60bd9a0530865110c2310f62a896c73350fa383c7812d5c6733e376bc377d`.

All 852,991,040 parameters were marked trainable. Uniform five-class
cross-entropy, AdamW, batch size 4, gradient accumulation 8, effective batch
32, learning rate 2e-5, weight decay 0.01, cosine scheduling, 5% warmup (11
steps), clipping 1.0, three epochs, and 213 updates are unchanged. The gradient
audit records gradients for the language backbone and classifier head:
752,398,144 parameters had gradients. The 100,592,896 vision parameters had
none because the path is text-only.

Training took 1224.5802961867303 seconds. Peak allocated/reserved training GPU
memory was 9.0316162109375/9.6015625 GiB and recorded RSS was
1.9677543640136719 GiB. Selected-checkpoint loading took
2.7541816290467978 seconds and isolated 924-example inference took
34.72815803065896 seconds. Final-validation peak allocated/reserved GPU memory
was 1.7161517143249512/1.77734375 GiB with RSS 1.7001190185546875 GiB.
Local retrieval took 151.0063940999098 seconds; known training, final
validation, and retrieval duration is 1413.0690299463458 seconds.

The selected checkpoint produces 924/924 valid outputs. Final
accuracy/macro-F1/weighted F1/balanced accuracy/top-2 accuracy is
0.36904761904761907/0.22994524079282935/0.3644656337102369/
0.2391812745015638/0.6341991341991342. It predicts
0/16/219/332/357 against Oracle support 13/81/178/232/420. The class-160
majority remains 420/924 = 45.45%, class 10 remains unpredicted with zero
recall, and class-20 recall is 0.024691358024691357. The exact-token prompt
therefore does not remove the imbalance.

Unchanged same-paper `top_k=5` retrieval covers 924/924 examples with
mean/median joined retrieval F1
0.2767166677489178/0.2558975. These are downstream token-overlap metrics, not
Oracle-label classification metrics. Relative to Phase 2C, Phase 2D improves
accuracy, macro-F1, weighted F1, and balanced accuracy, while mean joined
retrieval F1 decreases by 0.0024305281385280653. This is a clean prompt-only
comparison but remains a single seed selected and reported on validation.

Recorded CUDA-host commands using the separate RAM-backed Phase 2D root:

```bash
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle inspect
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle train --mode full --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle final-validation --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
```

Canonical predictions were copied back and evaluated through the unchanged
local Qdrant service. The retrieval evaluator is read-only, paper-restricted,
and uses the frozen `top_k=5` joined-F1 protocol. Local commands are:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py evaluate-retrieval --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py compare --output outputs\qwen_phase2d_comparison_evidence_length_oracle\six_way_comparison.json
```

The selected `step-000213` checkpoint was transferred from Vast.ai instance
46617164. The independently hashed 2,886,773,596-byte archive has SHA-256
`2dd4d23ff77179e1b33e522829cb2fdd6dd12684500a2158cc95f5f79a242a56`;
all nine extracted checkpoint-file hashes match the remote source. The
original Python 3.9 `.venv`, system Python, prior Qwen artifacts, Qdrant
collections, records, port, and storage path remained unchanged.

## Phase 2E remote CUDA environment

Phase 2E reuses the Phase 2D token-count prompt and five-logit Base-model
classifier while varying the learning rate across three fresh, independent
five-epoch runs. It uses the same remote executable,
`/workspace/thesis-granularity-router/.venv-qwen/bin/python`, with exact Python
version `3.10.7 (main, Oct  3 2022, 02:19:58) [Clang 14.0.3 ]`. The recorded
software and hardware provenance is unchanged:

- Transformers `5.15.0.dev0`, commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`;
- PyTorch `2.8.0+cu128`, CUDA 12.8, TensorBoard `2.20.0`;
- one `NVIDIA A100-SXM4-40GB` on `cuda`;
- `torch.bfloat16`, no quantization, deterministic seed 42.

The study ID is
`qwen-phase2e-lr-grid-token-count-prompt-5epochs-seed42-v1` and its
formulation is
`qwen-phase2e-base-sequence-classifier-token-count-prompt-lr-grid-v1`.
Run IDs are:

- `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1`;
- `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr1e-5-5epochs-full-parameter-20260808-seed42-v1`;
- `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr2e-5-5epochs-full-parameter-20260808-seed42-v1`.

Each trial loads `Qwen/Qwen3.5-0.8B-Base` revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` independently through
`AutoModelForSequenceClassification`. Each begins from the same initial
classifier-head hash,
`09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5`;
no trial continues from Phase 2D or from another grid trial. Five logits map
class IDs 0--4 to 10/20/40/80/160. The winning trained head hash is
`eb2cdb99b95c6941967fa9ec772729fd27c6ae613ffd9a7215332e0ede39b933`.

The exact frozen instruction is:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160 tokens. Return only the number

Its SHA-256 is
`b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5`.
The input template remains
`{instruction}\n\nQuestion: {original_question_text}`. Inputs contain no
evidence, evidence length, answer, paper text, retrieved chunks or scores,
embeddings, metadata, or handcrafted features. Train lengths are 95--121
tokens and validation lengths are 96--124; none is truncated at maximum length
128.

The three learning rates are 5e-6, 1e-5, and 2e-5. All other within-grid
training settings are fixed: full-parameter uniform cross-entropy, AdamW,
batch size 4, gradient accumulation 8, effective batch size 32, weight decay
0.01, cosine scheduling, warmup ratio 0.05, gradient clipping 1.0, five epochs,
and 355 updates. Warmup is 18 steps, with validation checkpoints at steps 71,
142, 213, 284, and 355. All 852,991,040 parameters are marked trainable;
752,398,144 language/classifier parameters receive gradients, while
100,592,896 vision parameters receive none on the text-only path.

The preserved data are 2,245 training examples from 845 papers and 924
validation examples from 277 papers. Training support is
55/267/586/687/650 and validation support is 13/81/178/232/420 for
10/20/40/80/160. Frozen Oracle SHA-256 values are
`64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`
for train and
`ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`
for validation.

The fixed global checkpoint-selection rule ranks all 15 checkpoints by higher
validation macro-F1, accuracy, weighted F1, and balanced accuracy; lower
validation cross-entropy; earlier step; and finally lower numeric learning
rate. It selected the 5e-6 trial at epoch 4, step 284. Selection was locked
before downstream retrieval, and retrieval cannot select or revise it. The
grid fingerprint is
`dc80671e8635cb2e479c7e231662eedb1be0920e28497d8f8e8b016703ff2b2b`.

Recorded training wall times for 5e-6, 1e-5, and 2e-5 are respectively
2044.1943467836827, 2022.7333836276084, and 2067.4948720689863 seconds;
their sum is 6134.422602480277 seconds. Each trial recorded peak allocated and
reserved GPU memory of 9.0316162109375 and 9.62109375 GiB. The selected final
validation load and inference times are 2.47247052565217 and
33.506004774942994 seconds. Mean and median inference latency are
0.03612477649043584 and 0.035192497074604034 seconds/question. The selected
final-validation peak is 1.7161517143249512 GiB allocated, 1.77734375 GiB
reserved, and 1.6993751525878906 GiB RSS. Known grid training plus selected
loading and inference time is 6170.401077780873 seconds.

The selected checkpoint produces 924/924 valid outputs. Its
accuracy/macro-F1/weighted F1/balanced accuracy/top-2 accuracy is
0.3484848484848485/0.22777929657889012/0.3473258648868964/
0.24232226137689133/0.6190476190476191, with prediction distribution
0/15/275/366/268. These are development-set classification results. The 924
examples are repeatedly observed for checkpoint and hyperparameter selection,
so they are not an unbiased final generalization estimate. This is one seed;
the three learning-rate trials are not seed replicates, no QASPER test result
is claimed, and no run-to-run variance claim is supported.

Phase 2E is not a pure learning-rate ablation against Phase 2D: Phase 2D used
three epochs, 213 steps, and 11 warmup steps, whereas Phase 2E uses five
epochs, 355 steps, and 18 warmup steps. The epoch count and cosine-schedule
horizon therefore change alongside the grid search.

Phase 2E same-paper, read-only `top_k=5` retrieval completed for all 924
selected-trial predictions: coverage is 924/924 = 1.0. Mean and median joined
retrieval F1 are 0.2793735097402597 and 0.267412. Because every classifier
output is valid, the coverage-adjusted full-set mean is
0.27937350974026. The uninterrupted one-segment retrieval wall time is
282.3799051999813 seconds. The frozen retrieval evaluation-configuration hash
is `9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8`.
The 5e-6, epoch-4, `step-000284` classification winner remained locked and
unchanged; retrieval was not used to select or revise it.

Recorded CUDA-host commands:

```bash
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle prepare
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle inspect --variant lr5e-6
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle train --variant lr5e-6
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle inspect --variant lr1e-5
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle train --variant lr1e-5
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle inspect --variant lr2e-5
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle train --variant lr2e-5
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle select
.venv-qwen/bin/python qwen_phase2e_sequence_classifier_lr_grid.py --study-root /dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle final-selected
```

Recorded local read-only retrieval commands are:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root outputs\qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle audit-selected
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root outputs\qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle retrieve-selected
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root outputs\qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle audit-final
```

## Phase 1 records

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

## Phase 2 records

- Direct dependencies: `requirements-qwen-phase2.txt`.
- Exact environment lock before/after execution:
  `outputs/qwen_finetuned_router_evidence_length_oracle/environment/phase2_package_lock.txt`
  and `phase2_package_lock_after.txt`.
- Hardware and CUDA snapshots:
  `outputs/qwen_finetuned_router_evidence_length_oracle/environment/vast_hardware_before_phase2.json`,
  `nvidia_smi_before_phase2.txt`, and `nvidia_smi_after_training.txt`.
- Training script snapshots and hashes:
  `outputs/qwen_finetuned_router_evidence_length_oracle/environment/qwen_phase2_training_launch.py`,
  `qwen_phase2_posttraining.py`, and `phase2_script_hashes.txt`.
- Final summary and integrity audit:
  `outputs/qwen_finetuned_router_evidence_length_oracle/final_summary.json` and
  `integrity_audit.json`.
- Exact run configuration, histories, checkpoint metadata, TensorBoard audit,
  and checkpoint verification:
  `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/`.
- Phase 2 reports: `docs/QWEN_PHASE2_RESULTS.md` and
  `reports/qwen_finetuned_router_evidence_length_oracle/experiment_report.md`.

## Phase 2B records

For each Phase 2B root, the authoritative environment and configuration
records are `configuration/experiment.json`,
`configuration/preflight_manifest.json`, the selected run's
`training_config.json`, `dataset_manifest.json`,
`formatted_example_inspection.json`, histories, checkpoint manifest, and
`summary.json`. The final summaries are:

- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/final_summary.json`;
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/final_summary.json`.

Each root also contains canonical validation predictions and runtime,
classification metrics/matrix/histogram, and retrieval records/runtime/summary.
The selected checkpoint and TensorBoard events are under the run-specific
`checkpoints/` and `tensorboard/` paths; large model files may remain
Git-ignored while retained locally. The machine-readable four-way comparison
is
`outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json`.
Human-readable records are `docs/QWEN_PHASE2B_RESULTS.md` and the two reports
under `reports/qwen_phase2b_alias_unweighted_evidence_length_oracle/` and
`reports/qwen_phase2b_alias_classbalanced_evidence_length_oracle/`.

## Phase 2C records

- Experiment and preflight configuration:
  `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/configuration/experiment.json`
  and `configuration/preflight_manifest.json`.
- Final summary:
  `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/final_summary.json`.
- Run configuration, dataset manifest, formatted-example inspection,
  gradient-coverage audit, histories, checkpoint manifest, and selected
  checkpoint metadata under
  `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/runs/qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1/`.
- Canonical predictions/raw outputs/parsed predictions/invalid records/runtime
  under `validation/`; metrics, confusion matrix, and histogram under
  `classification/`; retrieval records, segments, and summary under
  `retrieval/`.
- Five-way machine-readable comparison:
  `outputs/qwen_phase2c_comparison_evidence_length_oracle/five_way_comparison.json`.
- Human-readable reports: `docs/QWEN_PHASE2C_RESULTS.md` and
  `reports/qwen_phase2c_sequence_classifier_evidence_length_oracle/experiment_report.md`.

## Phase 2D records

- Experiment marker and preflight audit:
  `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/configuration/experiment.json`
  and `configuration/preflight_manifest.json`.
- Authoritative final summary:
  `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/final_summary.json`.
- Run configuration, dataset manifest, formatted-example inspection,
  gradient-coverage audit, training/validation histories, checkpoint manifest,
  selected-checkpoint metadata, TensorBoard events, and the locally retained
  checkpoint under
  `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/runs/qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1/`.
- Canonical predictions, raw logits, parsed predictions, the empty invalid
  record, and final runtime under `validation/`; metrics, confusion matrix,
  and predicted-vs-Oracle histogram under `classification/`; durable records,
  runtime segments, and summary under `retrieval/`.
- Remote/local selected-checkpoint verification:
  `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/integrity/selected_checkpoint_transfer_verification.json`.
- Independent 73-assertion artifact/Qdrant/TensorBoard audit and recorded
  102-test focused regression result:
  `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/integrity/final_integrity_audit.json`.
- Six-way comparison and Phase 2C-to-Phase 2D prompt-only protocol audit:
  `outputs/qwen_phase2d_comparison_evidence_length_oracle/six_way_comparison.json`.
- Reproduction implementations and focused regression tests:
  `qwen_phase2d_sequence_classifier.py`, `qwen_phase2d_posttraining.py`,
  `tests/test_qwen_phase2d_sequence_classifier.py`, and
  `tests/test_qwen_phase2d_posttraining.py`.

Large selected-checkpoint and TensorBoard files remain local/Git-ignored where
configured; the lightweight provenance, predictions, metrics, retrieval
records, and integrity manifests are the commit-oriented records. Phase 2D is
separate from Phase 2C and does not overwrite it.

## Phase 2E records

- Grid configuration and experiment identity:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/configuration/grid_experiment.json`.
- Global winner identity and selection audit:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/comparison/selected_trial.json`.
- All 15 checkpoint metrics:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/comparison/lr_grid_metrics.csv`.
- Selected final classification/runtime summary:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/comparison/selected_final_summary.json`.
- Trial-specific configurations, manifests, histories, validation outputs,
  classification results, and selected checkpoints:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/trials/lr5e-6/`,
  `trials/lr1e-5/`, and `trials/lr2e-5/`.
- TensorBoard event files remain with the retained remote originals under the
  corresponding `/dev/shm/.../trials/<variant>/tensorboard/` paths. They were
  deliberately excluded from the promoted local study; JSONL training and
  validation histories preserve the numeric training record locally.
- Canonical selected-trial summary, predictions, logits, parsed predictions,
  invalid record, runtime, metrics, matrix, and histogram:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/trials/lr5e-6/final_summary.json`,
  `validation/`, and `classification/`.
- Selected-checkpoint transfer verification and transfer manifests:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/integrity/selected_checkpoints_transfer_verification.json`
  and `integrity/transfer_manifests/`.
- Final post-retrieval integrity audit:
  `outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/integrity/final_post_retrieval_audit.json`.
  The audit records 64/64 metadata files verified at transfer time. After
  retrieval, 62 remain byte-identical and exactly two authorized summary
  rewrites are present: `comparison/selected_final_summary.json` and
  `trials/lr5e-6/final_summary.json`. It verifies all 13/13 transfer-bundle
  manifest files, all 27/27 files across the three retained checkpoints, zero
  forbidden payloads, and independently recomputes all 924 retrieval records.
- Reproduction implementations and focused tests:
  `qwen_phase2e_sequence_classifier_lr_grid.py`,
  `qwen_phase2e_posttraining.py`,
  `tests/test_qwen_phase2e_sequence_classifier_lr_grid.py`, and
  `tests/test_qwen_phase2e_posttraining.py`.
- Human-readable records: `docs/QWEN_PHASE2E_RESULTS.md` and
  `reports/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/experiment_report.md`.

Retrieval records are stored only below the locked winner at
`outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/trials/lr5e-6/retrieval/`:
`results.jsonl`, `runtime_segments.jsonl`, and `summary.json`. The selected
trial's `final_summary.json` and
`comparison/selected_final_summary.json` also contain the completed retrieval
summary. Phase 2E is separate from Phase 2D and does not overwrite any
previous experiment.
