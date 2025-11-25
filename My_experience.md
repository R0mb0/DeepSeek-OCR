# ⚙️ DeepSeek‑OCR: Technical Community Summary (final, English, emoji‑friendly)

A compact, technical and shareable summary of the work performed to run DeepSeek‑OCR locally (diagnostics, fixes, and practical next steps).

---

## 🚀 Quick overview
- Goal: run DeepSeek‑OCR (multimodal OCR) locally using the laptop GPU.
- Constraint: NVIDIA GeForce RTX 3060 Laptop GPU (~6 GiB VRAM).
- Main problems encountered:
  - package manager / C++ ABI conflicts (mamba / libmamba segfaults),
  - tokenizer + fork/thread init races (deadlocks),
  - tqdm/huggingface_hub compatibility (TypeError via tqdm.asyncio),
  - vllm / PyTorch OOM when loading large multimodal model.
- Current state: reproducible Conda env (Python 3.12) with PyTorch + CUDA 13 (installed via pip wheels), multiple launcher patches to avoid deadlocks and reduce footprint, monkeypatch to avoid tqdm.asyncio bug, diagnostics tooling added. Model still hits OOM on 6 GiB GPU in the heaviest phases unless quantized / run on a larger GPU.

---

## 🖥️ Hardware & OS (technical)
- GPU: NVIDIA GeForce RTX 3060 (Laptop) — ≈6 GiB VRAM.
- Driver: up to date for CUDA 12/13 runtimes.
- OS: Linux (Anaconda/conda available).
- RAM: laptop class (≥16 GiB recommended for substantial host spill).

---

## 🧰 Environment preparation — what was done
1. Created conda env (python 3.12) and used PyTorch official wheels for CUDA 13:
   - conda create -n torch13 python=3.12 pip -y
   - python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio

2. Verified GPU access with:
```py
import torch
print(torch.__version__, torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

3. Ensured critical environment variables are exported before importing tokenizers/torch/vllm:
```bash
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
```

---

## 🐛 Problems encountered (concise recap)
1. mamba / libmamba segmentation faults (package resolution) due to ABI mismatch (libstdc++) and duplicated/incompatible pkgs in conda cache.  
2. Tokenizer/fork+thread deadlocks (some modules create threads on import; fork + threads can deadlock).  
3. tqdm.asyncio mismatch: huggingface_hub/snapshot_download used tqdm.asyncio incorrectly in our environment leading to TypeError.  
4. vllm / PyTorch OOMs: heavy contiguous allocations during model load (packing/flatten) exceed 6 GiB.

---

## 🔧 Fixes applied (what to keep in repo)

### A — mamba / libstdc++ stability
- Actions:
  - Backup & remove problematic pkgs; clean conda caches.
  - Align libstdc++ from conda-forge:
    - conda remove -n base mamba libmamba libmambapy conda-libmamba-solver -y
    - conda clean --all -y
    - conda install -n base -c conda-forge libstdcxx-ng -y
    - conda install -n base -c conda-forge mamba -y
- Rationale: fix segfaults caused by mismatched shared libraries.

### B — Avoid tokenizer + fork deadlocks
- Launcher pattern (applied in patched launchers):
  - Set env vars before ANY heavy import.
  - Force multiprocessing start method to spawn:
```python
import os, multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_USE_V1'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
multiprocessing.set_start_method('spawn', force=True)
```
- Defer tokenizer init where possible (use skip_tokenizer_init in vllm engine args).

Files updated with this pattern: run_one_debug_skip_tokenizer.py, run_dpsk_ocr_image.py, run_dpsk_ocr_eval_batch.py, run_dpsk_ocr_pdf.py.

### C — tqdm.asyncio incompatibility (quick, robust fix)
- Problem: huggingface_hub snapshot_download called tqdm_class which resolved to tqdm.asyncio.tqdm_asyncio, and the constructor received duplicate `disable` kwarg (TypeError).
- Fix applied (two parts):
  1. A small wrapper and monkeypatch for local runs:
     - run_with_tqdm_patch.py — applies runtime monkeypatch (tqdm.asyncio.tqdm_asyncio -> tqdm.tqdm) and then runs target script.
  2. A sitewide, child-process visible patch: sitecustomize.py placed in repo root and ensured by adding repo root to PYTHONPATH. sitecustomize executes at every Python interpreter start, ensuring the monkeypatch applies also inside vllm EngineCore subprocesses (child interpreters).
- Why: vllm spawns child processes; patch must be visible in child interpreters — sitecustomize is the correct place.

### D — Conservative vllm / launcher settings (reduce VRAM footprint)
- Example conservative args used in patched launchers:
```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    block_size=256,
    max_model_len=8192,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.05,  # very low on GPU
    swap_space=12,                # GiB host spill
    skip_tokenizer_init=True,
)
```
- config.py: lowered defaults used by launchers:
  - MIN_CROPS=1, MAX_CROPS reduced (e.g. 1–4)
  - MAX_CONCURRENCY = 1..4 conservative
  - NUM_WORKERS lowered
- Rationale: avoid immediate large allocations; allow vllm to spill tensors to host.

### E — Diagnostics & helper scripts added
- gather_vllm_diagnostics.sh — collects nvidia-smi, ps, lsof, /proc info, strace (short), gdb backtrace, py-spy (if present) and archives to /tmp.
- test_vllm_gpt2.py — small test script to validate vllm engine on small model (gpt2) to separate infra issues from heavy model issues.
- run_with_tqdm_patch.py and sitecustomize.py (described above).

---

## 🔬 Diagnostics observations
- strace on EngineCore showed many threads in futex / epoll_wait: engine was waiting on IPC before fix; after tqdm patch the engine advanced further into model load.
- lsof showed vllm / torch shared libs loaded and GPU device file descriptors open.
- The final blocking point observed repeatedly: heavy weight load & tensor packing (vllm model loader) triggered large contiguous allocations that exceeded available VRAM → OutOfMemoryError inside the vllm worker.

---

## 📦 Quantization & recommended next steps (practical)
1. Best long‑term local solution on 6 GiB: quantize weights to 8‑bit (bitsandbytes). Typical VRAM reduction ~2–4×.
   - pip install bitsandbytes (choose compatible version for your CUDA runtime).
   - Integrate quantization with loader / vllm, e.g. set engine_args.quantization = QuantizationConfig(bits=8) if your vllm supports it, or load via transformers with load_in_8bit/device_map then adapt to vllm.
2. If quantization is not feasible: use a GPU with ≥12–24 GiB VRAM (cloud or remote machine).
3. For further local attempts: try extremely conservative settings (gpu_memory_utilization=0.01, swap_space=16 GiB, MAX_CROPS=1, MAX_CONCURRENCY=1) — may be extremely slow and still fail, but useful to test end‑to‑end flow.
4. Pre‑download model snapshot locally (huggingface snapshot_download) to avoid time/IO spikes during child process startup.

---

## ✅ Final status & actionable summary
- Environment: Conda env (Python 3.12) with PyTorch 2.9.x+cu130 installed via pip — PyTorch sees CUDA and the GPU.
- Code changes: patched launchers, conservative config, sitecustomize + run_with_tqdm_patch to fix tqdm/huggingface_hub mismatch.
- Diagnostics tooling: gather_vllm_diagnostics.sh and test_vllm_gpt2.py added.
- Remaining blocker: large multimodal model (DeepSeek‑OCR) needs quantization or larger GPU to reliably finish full initialization and inference on a 6 GiB device.

---

## 🧾 Useful copy‑paste commands

- Kill currently stuck vllm processes (use in another terminal):
```bash
PIDS=$(ps -eo pid,cmd | egrep 'VLLM::EngineCore|run_one_debug_skip_tokenizer|run_dpsk_ocr|run_one_debug|vllm' | awk '{print $1}' | tr '\n' ' ')
[ -n "$PIDS" ] && kill $PIDS
sleep 3
[ -n "$PIDS" ] && sudo kill -9 $PIDS || true
```

- Run diagnostics collector (in another terminal):
```bash
bash ./gather_vllm_diagnostics.sh
# archive will be in /tmp/vllm_diag_<timestamp>.tar.gz
```

- Quick PyTorch check:
```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

- Launch patched image pipeline (example):
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"   # ensures sitecustomize.py is loaded by child processes
python run_with_tqdm_patch.py run_dpsk_ocr_image.py
# or for single image:
python run_with_tqdm_patch.py run_one_debug_skip_tokenizer.py /path/to/image.png
```

---

## 📁 Files changed / added (high level)
- Patched launchers: run_one_debug_skip_tokenizer.py, run_dpsk_ocr_image.py, run_dpsk_ocr_eval_batch.py, run_dpsk_ocr_pdf.py (set envs early, spawn start method, conservative engine args).
- Config change: config.py (conservative defaults; note tokenizer is still created eagerly — see note below).
- New helper/diagnostic scripts: run_with_tqdm_patch.py, sitecustomize.py, gather_vllm_diagnostics.sh, test_vllm_gpt2.py.
- Model code: deepseek_ocr.py and associated modules left functionally intact but recognized as heavy for 6 GiB — quantization hooks available (QuantizationConfig import present).

---

## ⚠️ Notes & caveats
- Tokenizer is currently created eagerly inside config.py (AutoTokenizer.from_pretrained). For maximum safety, consider turning that into a lazy loader (get_tokenizer performs cache-lazy loading) to avoid I/O or thread creation at import time.
- sitecustomize.py must be found by Python in every child interpreter: ensure repo root is in PYTHONPATH before launching any patched script.
- When installing bitsandbytes or other CUDA-sensitive packages, choose versions compatible with your system CUDA and the wheel builds you use.
