# DeepSeek-OCR — Repository overview & file guide

This README describes the files and folders in this working copy of DeepSeek‑OCR, explains why each file was created, how each file is structured, and which other files it depends on at runtime. It also includes practical execution examples, environment notes, and troubleshooting tips (issues we encountered and the mitigations we added).

---

Table of contents
- Overview
- Quick start (recommended environment + key env vars)
- High-level architecture / data flow
- File-by-file description (what, why, structure, dependencies)
- How to run (examples)
- Troubleshooting & diagnostics
- Recommendations & next steps

---

Overview
========
This repo contains a multimodal OCR pipeline (DeepSeek‑OCR) that combines vision encoders and a language decoder (integrated with vllm) to produce markdown outputs and bounding‑box visualizations. The codebase was adjusted to be robust on resource‑constrained local machines (e.g. laptop RTX 3060 with ~6 GiB VRAM) by:

- setting environment variables early to avoid tokenizer/CUDA deadlocks;
- adding conservative defaults (lower concurrency / cropping);
- adding a monkeypatch/sitecustomize to work around a tqdm/huggingface_hub incompatibility in some environments;
- providing debug/run wrappers and small test scripts.

This README documents the files added/edited during the troubleshooting and development cycle.

---

Quick start — environment & important environment variables
============================================================
Recommended approach (tested):
1. Create a conda environment (python 3.12) and install PyTorch + CUDA 13 via official wheels:
   ```bash
   conda create -n torch13 python=3.12 pip -y
   conda activate torch13
   python -m pip install --upgrade pip
   python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
       torch torchvision torchaudio
   ```

2. Always set the following environment variables before importing or running any Python that touches tokenizers/torch/vllm:
   ```bash
   export TOKENIZERS_PARALLELISM=false
   export VLLM_USE_V1=1             # keep consistent with the vllm engine use in the code
   export CUDA_VISIBLE_DEVICES=0    # or set to the GPU index you want to use
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
   export PYTHONPATH="$PWD:$PYTHONPATH"   # ensures sitecustomize.py in repo root is loaded by child processes
   ```

Notes:
- The code expects `sitecustomize.py` (monkeypatch for tqdm.asyncio) to be importable by subprocesses. Putting repo root in PYTHONPATH ensures child interpreters see it.
- On memory-constrained GPUs, also lower runtime parameters in `config.py` (see below).

---

High-level architecture / runtime flow
======================================
- Image input → preprocessing (tile / crop / normalization) via `process/image_process.py` → image tokens and pixel tensors
- Multi‑modal processor (`vllm` integration) merges image embeddings and text inputs → produces model inputs for language decoder
- Inference performed by vllm engine (either AsyncLLMEngine or LLM sync client), which uses model class in `deepseek_ocr.py`
- Outputs: markdown text files, optionally annotated images with bounding boxes, and diagnostic logs

Main runtime entrypoints:
- `run_dpsk_ocr_image.py` — single-image or directory pipeline using AsyncLLMEngine (async)
- `run_dpsk_ocr_eval_batch.py` — batch evaluation flow using vllm LLM (sync) with conservative engine settings
- `run_dpsk_ocr_pdf.py` — PDF → images → inference pipeline, saves annotated PDF
- `run_one_debug_skip_tokenizer.py` — minimal debug script designed to create engine with skip_tokenizer_init=True and run a single image request
- `test_vllm_gpt2.py` — small test script that validates vllm + GPU works using the small `gpt2` model

---

Files & folders — detailed guide
================================

Top-level folders
-----------------
- `deepencoder/`  
  Contains vision model implementations (SAM / CLIP variants and helpers). Used by `deepseek_ocr.py` to build the visual feature extractors. Key modules used at runtime:
  - `deepencoder/sam_vary_sdpa.py` — SAM ViT patch & embedding code used to compute patch embeddings.
  - `deepencoder/clip_sdpa.py` — CLIP-style vision transformer code used as visual backbone.
  - `deepencoder/build_linear.py` — projector / MLP code used to map vision features to embedding dimension.

- `deepseek/`  
  (If present) may contain the original hf model code, packaging, or wrapper scripts; treat as project-specific utilities.

- `process/`  
  Preprocessing & generation helpers used by launchers:
  - `process/image_process.py` — main image preprocessing (cropping logic, tokenization with image tokens, transforms). Provides `DeepseekOCRProcessor` (registered to transformers/AutoProcessor) and helper functions `dynamic_preprocess` and `count_tiles`.
  - `process/ngram_norepeat.py` — custom logits processor `NoRepeatNGramLogitsProcessor` that forbids repetition of n‑grams in generation (used as `logits_processors` in sampling params).

Other important files
---------------------
- `config.py`  
  Purpose: central configuration used by launchers and processors; also creates a (currently eager) tokenizer via `AutoTokenizer.from_pretrained(MODEL_PATH)`.  
  - Why created: consolidate default operational values (image sizes, cropping defaults, concurrency, model path, prompt).
  - Structure: constants (IMAGE_SIZE, MAX_CROPS, MAX_CONCURRENCY, MODEL_PATH, PROMPT), then `TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)` and `get_tokenizer()` wrapper.
  - Important note: The tokenizer is created eagerly at import time which can trigger I/O or thread creation. Many fixes in this repo rely on *setting specific environment variables before importing `config`*; see the top of launchers where `TOKENIZERS_PARALLELISM` and `multiprocessing.set_start_method('spawn')` are set.

- `deepseek_ocr.py`  
  Purpose: the inference-only model wrapper wired into vllm's multimodal APIs.  
  - Why created/modified: adapt HF config/weights into vllm's model loader and implement image→embedding flows (calls into `deepencoder` modules).  
  - Structure: defines `DeepseekOCRForCausalLM` (a vllm-registered model), multimodal processing info & a processor class, `_pixel_values_to_embedding()` which composes SAM and CLIP features, and `load_weights()` mapping.  
  - Dependencies: `vllm`, `vllm.multimodal` APIs, `deepencoder` modules, `process.image_process.DeepseekOCRProcessor`, `config.get_tokenizer`.

- `run_dpsk_ocr_image.py`  
  Purpose: Async launcher to process a single image or directory of images using `AsyncLLMEngine`.  
  - Why created: provide robust entrypoint with fixes to avoid fork+thread deadlocks, safe tokenizer handling (tokenize after engine creation), and conservative engine args to reduce OOM risk.  
  - Structure: sets envs early (VLLM_USE_V1, TOKENIZERS_PARALLELISM, OMP_NUM_THREADS), forces `multiprocessing.set_start_method('spawn')`, imports vllm asynchronously, registers model, creates engine with `skip_tokenizer_init=True`, then for each image tokenizes and streams generation.  
  - Inputs: `config.INPUT_PATH` (file or directory) and `config.PROMPT`.  
  - Outputs: markdown .mmd files (original & cleaned) and annotated images in `OUTPUT_PATH`.  
  - Uses: `process.image_process.DeepseekOCRProcessor`, `process.ngram_norepeat.NoRepeatNGramLogitsProcessor`, `deepseek_ocr.DeepseekOCRForCausalLM`.

- `run_dpsk_ocr_eval_batch.py`  
  Purpose: a synchronous batch runner that pre‑tokenizes images in a thread pool and sends a batch to `vllm.LLM.generate()`.  
  - Why created: evaluate many images at once (batch inference) with safeguards for memory (swap_space, reduced gpu_memory_utilization).  
  - Structure: early envs set; builds `LLM(...)` with conservative parameters; preprocess images in `ThreadPoolExecutor`; calls `llm.generate()` and writes outputs to disk.  
  - Dependencies: as above (processor, model). Uses `config.MAX_CONCURRENCY` and `NUM_WORKERS`.

- `run_dpsk_ocr_pdf.py`  
  Purpose: PDF processing launcher — converts PDF pages to high‑quality images, tokenizes each page, runs inference and produces annotated PDF with layouts.  
  - Why created: support document workflows with many pages.  
  - Structure: uses `fitz` to rasterize pages at configurable DPI, reuses processor and model, and saves outputs and annotated PDF.

- `run_one_debug_skip_tokenizer.py`  
  Purpose: small debug utility to create an AsyncLLMEngine with `skip_tokenizer_init=True` and run one request (useful for debugging engine startup independent of tokenizers).  
  - Why created: reproduce engine initialization issues / quick local tests.  
  - Input: path to a single image on the command line.  
  - Output: streamed generation to stdout.  
  - Useful when diagnosing engine vs. tokenizer vs. model weight problems.

- `run_with_tqdm_patch.py`  
  Purpose: lightweight wrapper that applies a monkeypatch so `tqdm.asyncio.tqdm_asyncio` is replaced with `tqdm.tqdm` before running target script.  
  - Why created: some combos of `tqdm` + `huggingface_hub` call the asyncio tqdm in a way that causes TypeError: duplicate `disable` kwarg. vllm spawns child processes, so a `sitecustomize.py` monkeypatch is also required for children — see `sitecustomize.py`.  
  - Usage:
    ```bash
    python run_with_tqdm_patch.py test_vllm_gpt2.py
    python run_with_tqdm_patch.py run_one_debug_skip_tokenizer.py /path/to/image.png
    ```

- `sitecustomize.py`  
  Purpose: executed at Python interpreter startup when the repo root is in `PYTHONPATH`. It applies the same monkeypatch in every interpreter (including vllm child processes).  
  - Why created: vllm spawns child interpreters; the parent-level monkeypatch is not sufficient. `sitecustomize.py` guarantees patching occurs for children.  
  - Important: ensure `PYTHONPATH` contains repository root before launching any scripts.

- `test_vllm_gpt2.py`  
  Purpose: sanity test that creates a small `LLM`/Engine using `gpt2` to ensure vllm + GPU works in the environment without the heavy DeepSeek model.  
  - Why created: separate failure modes (environment vs. heavy model) and reduce time-to-diagnose.

- `ngram_norepeat.py` (`process/ngram_norepeat.py`)  
  Purpose: custom logits processor which bans repeating n‑grams in generated outputs, with a configurable window and whitelist token ids.  
  - Why created: to improve textual stability for OCR tagging and avoid repeated markup tokens in output.  
  - Used by: sampling params in run scripts.

- `image_process.py` (`process/image_process.py`)  
  Purpose: image preprocessing and the `DeepseekOCRProcessor` which tokenizes prompts and attaches image tokens / cropped image tensors.  
  - Important: `tokenize_with_images()` depends on a tokenizer being available via `config.get_tokenizer()` — if that tokenizer is `None`, calls will raise. Launchers use `skip_tokenizer_init=True` for engine cores to avoid double init and then call processor after creating engine.

- `deepencoder/*`  
  Purpose: vision backbones (ViT/SAM/CLIP) and projection heads used by `deepseek_ocr.py` to produce embeddings. These modules are generally heavy and the main place where runtime memory is allocated when computing patch embeddings.

- `run_quantized_debug.py`, `run_quantized_engine_test.py`, `scripts_test_bnb_load.py`, `scripts_test_bnb_load_shim.py`, `try_awq_quant_engine_auto.py`, `scripts_*`  
  Purpose: experiments and helper scripts to test quantization (`bitsandbytes`, AWQ) and alternative loading strategies. These are experimental utilities used when attempting to run the model on small GPUs (6 GiB) using 8/4-bit weight formats; they are not required to run the core pipeline but are useful for follow-up work.

- Various `scripts_*` and `tests` (`test_model_access.py`, `scripts_import_*`, `scripts_save_model_states.py`, etc.)  
  Purpose: utility scripts used to prepare, cache, and remap weights for local testing, and to verify that loader/weight mapping works as expected.

---

How files interact at runtime (dependency map)
----------------------------------------------
- Launchers (`run_*.py`) → import `config` and read constants (MODEL_PATH, PROMPT, INPUT_PATH, OUTPUT_PATH)  
- Launchers → import `deepseek_ocr.DeepseekOCRForCausalLM` (ModelRegistry.register_model called in launchers)  
- `deepseek_ocr.py` → imports `deepencoder/*` (vision backbones), `process/image_process.DeepseekOCRProcessor` and `config.get_tokenizer`  
- `process/image_process.py` → uses `config` for sizes and tokenizer via `get_tokenizer()`  
- Sampling/logits processors: `process/ngram_norepeat.py` used by launchers in `SamplingParams`  
- Helpers (quantization scripts) may import `vllm.model_executor.layers.quantization` or `bitsandbytes` if doing 8‑bit loading experiments

Practical execution
-------------------
1. Ensure environment & env vars:
   ```bash
   conda activate torch13
   export TOKENIZERS_PARALLELISM=false
   export VLLM_USE_V1=1
   export CUDA_VISIBLE_DEVICES=0
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
   export PYTHONPATH="$PWD:$PYTHONPATH"
   ```

2. Use `run_with_tqdm_patch.py` wrapper if your local `tqdm` + `huggingface_hub` combo triggers the TypeError:
   ```bash
   python run_with_tqdm_patch.py run_one_debug_skip_tokenizer.py /path/to/image.png
   # or
   python run_with_tqdm_patch.py run_dpsk_ocr_image.py
   ```

3. Run batch or pdf pipelines (example):
   ```bash
   python run_with_tqdm_patch.py run_dpsk_ocr_eval_batch.py
   python run_with_tqdm_patch.py run_dpsk_ocr_pdf.py
   ```

4. Quick engine debug (small gpt2 test):
   ```bash
   python run_with_tqdm_patch.py test_vllm_gpt2.py
   ```

Notes:
- For `run_dpsk_ocr_image.py` the script reads `config.INPUT_PATH` — update `config.py` or call the launcher directly (if refactored to accept CLI args).
- For `run_one_debug_skip_tokenizer.py` pass an image path on the command line.

Troubleshooting & known issues
------------------------------
1. tqdm / huggingface_hub TypeError
   - Symptom: TypeError: tqdm.asyncio.tqdm_asyncio.__init__() got multiple values for keyword argument 'disable'
   - Mitigation: run `run_with_tqdm_patch.py` or ensure `sitecustomize.py` is on `PYTHONPATH`. This repository includes `run_with_tqdm_patch.py` and `sitecustomize.py` to patch `tqdm.asyncio.tqdm_asyncio` → `tqdm.tqdm`.

2. Tokenizer / fork+thread deadlocks
   - Symptom: child processes hang during engine init, with futex/epoll_wait stack traces.
   - Mitigation: all launchers set the following before any heavy import:
     ```python
     os.environ['TOKENIZERS_PARALLELISM']='false'
     multiprocessing.set_start_method('spawn', force=True)
     ```
     Also `skip_tokenizer_init=True` when creating vllm engine and tokenize images after engine creation where possible.

3. mamba / libmamba segmentation faults (environment-level)
   - Symptom: `mamba` segfaults during package resolution.
   - Mitigation: we backed up problematic packages in `~/anaconda3-pkgs-backup`, cleaned caches, installed `libstdcxx-ng` from conda‑forge and reinstalled mamba from conda‑forge. If you still see segfaults, consider reinstalling base conda or using pip wheels inside a conda environment as demonstrated in Quick Start.

4. vllm / PyTorch OOM on small GPU (primary blocking issue)
   - Symptom: OutOfMemoryError on model load or during packing/flatten operations.
   - Mitigations we added:
     - `PYTORCH_CUDA_ALLOC_CONF` to mitigate fragmentation
     - conservative runtime args: `gpu_memory_utilization` small, `swap_space` set to spill to host, `MAX_CONCURRENCY` & `MAX_CROPS` reduced
     - recommended long-term fix: convert to 8‑bit quantized weights (bitsandbytes / AWQ), or run on a machine with more VRAM (12–24 GiB)
   - Useful commands to observe GPU:
     ```bash
     watch -n 2 nvidia-smi
     nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv
     ```

Diagnostics & helper scripts
----------------------------
- `test_vllm_gpt2.py` — smoke test for vllm engine using `gpt2`
- `run_one_debug_skip_tokenizer.py` — single request engine debug (skip_tokenizer_init=True)
- `run_with_tqdm_patch.py` & `sitecustomize.py` — monkeypatch for tqdm/huggingface_hub issue
- `run_quantized_*`, `scripts_*` — tests & helper scripts for quantization & weight manipulation

Recommendations & next steps
----------------------------
- If you need to run DeepSeek‑OCR for real workloads on the laptop:
  - Try 8‑bit quantization paths (install `bitsandbytes` matching your CUDA, use provided `run_quantized_*` helpers). I can generate a patch that integrates quantization into the loader/engine args.
  - Otherwise move to an instance with ≥12 GiB VRAM for good performance.
- If you want reproducible environment artifacts: generate `env.yml` and `requirements.txt` from current env:
  ```bash
  pip freeze > ~/torch13-requirements.txt
  conda list -n torch13 > ~/torch13-conda-list.txt
  ```
- Consider converting the eager `TOKENIZER` creation in `config.py` to a lazy loader (e.g. only `get_tokenizer()` loads on demand) to avoid I/O at import time and reduce threading risks.
1) Generate `env.yml` + `requirements.txt` from the current environment,
2) Create `setup_complete.sh` to automate environment repairs + install,
3) Produce a patch to add 8‑bit quantization into the vllm loader / `deepseek_ocr.py`.
