#!/usr/bin/env python3
"""
Prova a creare una QuantizationConfig concreta (usando la factory di vllm)
e a costruire un AsyncLLMEngine quantizzato (create + shutdown).
Esegui con l'env attivo (conda activate deepseek-ocr).

Esempio:
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
python run_quantized_engine_test.py
"""

import os
import time
import traceback
import inspect

# --- set envs BEFORE importing vllm/torch/transformers ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --- imports after envs ---
try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.layers import quantization as qmod
    from vllm.model_executor.models.registry import ModelRegistry
except Exception as e:
    print("Import error (vllm):", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(1)

# local model + config
try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    from config import MODEL_PATH
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
except Exception as e:
    print("Import project modules error:", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(1)

print("quant module:", qmod.__file__)
print("Available QUANTIZATION_METHODS keys:", getattr(qmod, "QUANTIZATION_METHODS", None))
print("Signature of get_quantization_config:", inspect.signature(qmod.get_quantization_config))

# pick the method that in precedenti test ha funzionato ('awq')
method = "awq"
print(f"Attempting to obtain quantization config for method: '{method}'")

# get the config class (factory returns a class/type)
try:
    qc_cls = qmod.get_quantization_config(method)
    print("Returned qc class/type:", qc_cls)
    try:
        print("qc class signature:", inspect.signature(qc_cls))
    except Exception:
        pass
except Exception as e:
    print("get_quantization_config failed:", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(2)

# Try to instantiate the quant config with a few sensible fallbacks
qc = None
attempts = [
    lambda: qc_cls(),                       # no args
    lambda: qc_cls(8),                      # positional bits
    lambda: qc_cls(bits=8),                 # kw bits
    lambda: qc_cls(nbits=8),                # other kw
    lambda: qc_cls(config={"bits": 8}),     # config dict
]
for i, fn in enumerate(attempts, start=1):
    try:
        qc_try = fn()
        if qc_try is not None:
            qc = qc_try
            print(f"QuantizationConfig instantiated with attempt #{i}: {type(qc)}")
            break
    except TypeError as te:
        print(f"Attempt #{i} TypeError: {te}")
    except Exception as e:
        print(f"Attempt #{i} failed: {type(e).__name__}: {e}")

if qc is None:
    print("Could not instantiate QuantizationConfig automatically. Stopping.")
    raise SystemExit(3)

# Show a brief repr of qc (avoid overly long output)
try:
    print("repr(qc) (truncated):", repr(qc)[:400])
except Exception:
    pass

# Build conservative AsyncEngineArgs using this quant config
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    max_model_len=8192,
    enforce_eager=False,          # avoid extreme compilations for debug
    trust_remote_code=True,
    skip_tokenizer_init=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.2,   # conservative for small GPUs
    swap_space=8,                 # GiB
    quantization=qc,
)

print("Creating AsyncLLMEngine (quantized) with conservative settings...")
t0 = time.time()
try:
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Engine created in %.1f s" % (time.time() - t0))
except Exception as e:
    print("Engine creation FAILED:", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(4)

# shutdown right away (test only)
try:
    print("Shutting down engine...")
    engine.shutdown()
    print("Engine shutdown OK")
except Exception as e:
    print("Error during engine.shutdown():", type(e).__name__, e)
    traceback.print_exc()

print("Done.")