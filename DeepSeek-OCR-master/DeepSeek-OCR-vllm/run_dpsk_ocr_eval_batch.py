#!/usr/bin/env python3
"""
Batch evaluation launcher (patched).
Key fixes:
- Set environment variables BEFORE importing torch / vllm to avoid tokenizer/CUDA init races.
- Use conservative GPU settings (lower gpu_memory_utilization, enable swap_space) to reduce OOM on small GPUs.
- Small robustness improvements when listing/opening images.
"""
import os
import re
import glob
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# ENVIRONMENT (must be set early)
# -----------------------------
# Control tokenizers parallelism and vllm mode before importing torch/vllm
os.environ['TOKENIZERS_PARALLELISM'] = os.environ.get('TOKENIZERS_PARALLELISM', 'false')
os.environ['VLLM_USE_V1'] = os.environ.get('VLLM_USE_V1', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# Limit native threadpools to reduce threading at fork time
os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
os.environ['MKL_NUM_THREADS'] = os.environ.get('MKL_NUM_THREADS', '1')
# PyTorch allocator hint (help fragmentation) — can be overridden in shell as needed
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# Now safe to import torch and other heavy libs
import torch
from tqdm import tqdm
from PIL import Image

# If you use CUDA 11.8 and need ptxas path, keep this check after importing torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

# Project imports (after envs)
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, MAX_CONCURRENCY, CROP_MODE, NUM_WORKERS
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# Register model implementation
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# -----------------------------
# LLM construction (conservative)
# -----------------------------
# Lower gpu_memory_utilization and enable swap_space so the model can load on smaller GPUs.
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=8,                 # 8 GiB swap to host (vllm expects GiB)
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.12,     # conservative default to avoid OOM on small GPUs
    quantization="bitsandbytes",
    disable_mm_preprocessor_cache=False,
)

logits_processors = [
    NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822})
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# -----------------------------
# Helpers
#-------------
# (rest of file unchanged)