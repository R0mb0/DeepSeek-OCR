# ⚙️ DeepSeek‑OCR: Technical Community Summary (English, emoji‑friendly)

A compact, technical and shareable summary of the work performed to run DeepSeek‑OCR locally.

---

## 🚀 Quick overview
- Goal: Run DeepSeek‑OCR (multimodal model) locally with GPU acceleration.
- Key constraints: laptop GPU (~6 GiB VRAM), modern NVIDIA driver supporting CUDA 12/13.
- Main issues encountered: package manager/library ABI conflicts, tokenizer + fork/thread init races, and GPU OOM when loading the model.
- Current state: working PyTorch + CUDA environment; code patched to reduce memory footprint and avoid deadlocks; model still hits OOM on 6 GiB GPU in the most memory‑heavy phases (packing/flattening tensors). Options: more aggressive spilling, 8‑bit quantization, or larger GPU.

---

## 🖥️ Hardware & OS (anonymous, technical)
- GPU: NVIDIA GeForce RTX 3060 (Laptop) — ~6 GiB VRAM.
- NVIDIA driver: recent (compatible with CUDA 12/13 runtimes).
- Host OS: Linux (typical distro with conda/Anaconda available).
- RAM: typical laptop class (≥16 GiB recommended if planning to spill to host).

---

## 🧰 Environment preparation (what was done)
1. Created an isolated Conda environment (Python 3.12) and used pip wheels for exact PyTorch/CUDA:
   - Create env:
     - conda create -n torch13 python=3.12 pip -y
   - Install PyTorch CUDA builds (official PyTorch index):
     - python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
         torch torchvision torchaudio

2. Verified GPU access with PyTorch:
```python
import torch
print(torch.__version__, torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
```

3. Exported key environment variables _before_ importing tokenizers / CUDA libraries:
```bash
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1          # keep consistent with AsyncLLMEngine v1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
```

---

## 🐛 Problems encountered (concise)
1. mamba / libmamba segmentation faults during package resolution — caused by library ABI mismatch (libstdc++ / GLIBCXX) and duplicated/incompatible package files in the conda cache.  
2. Tokenizer / threads deadlock risk: some modules create threads on import; using fork() with threads can deadlock.  
3. vllm / PyTorch OOM during model initialization: large contiguous allocation (~100–200 MiB) fails while ~5 GiB is already allocated/reserved — model is too large for 6 GiB GPU without quantization or spilling.

---

## 🔧 How problems were solved — logical fixes & notable code snippets

### 1) mamba / libmamba ABI conflicts
- Approach: clean package cache and align libstdc++ to conda‑forge release.
- Commands (conceptual, adapt to your environment):
```bash
# Backup problematic packages
mkdir -p ~/conda-pkgs-backup
# Move problematic libmamba/mamba packages out of the cache, then:
conda remove -n base mamba libmamba libmambapy conda-libmamba-solver -y
conda clean --all -y
conda install -n base -c conda-forge libstdcxx-ng -y
conda install -n base -c conda-forge mamba -y
```
- Rationale: ensures the runtime C++ ABI used by conda tooling matches the installed system libraries.

---

### 2) Avoiding tokenizer fork+thread deadlocks
- Strategy: set environment variables BEFORE importing tokenizers/torch and use spawn start method for multiprocessing when needed.
- Example snippet to put at top of launcher scripts (very early):
```python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_USE_V1'] = os.environ.get('VLLM_USE_V1', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```
- Also: defer tokenizer initialization when possible (avoid eager tokenizer creation in imported modules).

---

### 3) Reducing GPU footprint / enabling spill to host (vllm)
- vllm supports controlling how much state stays on GPU and an explicit swap space on host (value in GiB).
- Conservative engine args used to attempt to load the model on small GPUs:
```python
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    max_model_len=8192,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.05,  # very small GPU retention
    swap_space=12,                # GiB of host spill space
    skip_tokenizer_init=True,
)
```
- Also reduced visual processing concurrency:
```python
# config.py conservative values
MAX_CROPS = 1
MAX_CONCURRENCY = 1
NUM_WORKERS = 1
```
- Note: swap_space is in GiB and uses host RAM/SSD; it reduces OOM risk but makes load & inference much slower.

---

### 4) Quantization (recommended for 6 GiB GPUs)
- Best long‑term solution: load model weights quantized to 8‑bit (bitsandbytes). Typical VRAM reduction: ~2–4×.
- Rough steps:
  - pip install bitsandbytes (ensure compatibility with your CUDA runtime)
  - configure vllm/loader with a quantization config (if supported), e.g.:
```python
from vllm.model_executor.layers.quantization import QuantizationConfig

engine_args.quantization = QuantizationConfig(bits=8)
```
- If vllm doesn't expose that path for your version, use transformers + bitsandbytes device_map/load_in_8bit approaches, then adapt to vllm or run inference with transformers directly.

---

## ✅ Current status (where I stand)
- Environment: Conda env with Python 3.12 and PyTorch wheel built for CUDA 13 installed via pip. PyTorch detects GPU successfully.
- Code: launchers patched to:
  - set env vars before imports,
  - use spawn start method,
  - defer tokenizer init (skip_tokenizer_init when creating vllm engine),
  - lower gpu_memory_utilization and add swap_space values for host spill.
- Outcome: vllm engine initializes further than before but still throws OOM on 6 GiB GPU during heavy tensor packing steps.
- Workable fallbacks:
  - Aggressive spilling (very slow) — may or may not succeed depending on host RAM and fragmentation patterns.
  - 8‑bit quantization (preferred) — requires bitsandbytes + loader adjustments.
  - Run on a larger GPU in cloud / remote machine (most straightforward to get full performance).

---

## 💡 Recommendations & next steps
1. Quick try: set gpu_memory_utilization = 0.05 and swap_space = 12 GiB, then run while monitoring `nvidia-smi`. This may allow model to load (slow).
2. Best practical solution for 6 GiB: enable 8‑bit quantization (install bitsandbytes and adapt loader). I can prepare exact code patches for this repo.
3. If speed is important and you can afford it: run inference on a machine with ≥16 GiB VRAM.
4. Always set these env vars before importing tokenizer/torch:
```bash
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
```

---

## 📦 Useful commands (copy‑friendly)

- Quick PyTorch GPU check:
```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

- Minimal vllm engine config snippet (for launchers):
```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    gpu_memory_utilization=0.05,
    swap_space=12,
    skip_tokenizer_init=True,
)
```

---

## 📝 TL;DR — one‑line verdict
If you want stable local runs on a 6 GiB GPU: prefer 8‑bit quantization (bitsandbytes) or accept very slow host swapping; otherwise use a larger GPU (cloud or local). I can prepare the quantization patch or a one‑shot setup script — say the word and I’ll generate it. 😄

---
If you'd like, I can now:
- provide a ready patch to enable bitsandbytes quantization in this repo, or
- produce a small bash script that applies the conservative engine/config changes automatically and runs a monitored test.
Which would you prefer? 🔧🧪
