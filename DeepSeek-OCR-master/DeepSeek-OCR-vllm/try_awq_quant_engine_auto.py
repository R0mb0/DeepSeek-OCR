#!/usr/bin/env python3
"""
Prova automaticamente alcune configurazioni AWQ (4-bit) e tenta di creare
un AsyncLLMEngine quantizzato (create + shutdown). Stampa dettagli e si ferma
alla prima combinazione funzionante.

Uso:
  conda activate deepseek-ocr
  export TOKENIZERS_PARALLELISM=false
  export VLLM_USE_V1=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
  python try_awq_quant_engine_auto.py
"""
import time, traceback, inspect, sys, os

# (opzionale) assicurati di avere env vars prima degli import pesanti
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", os.environ.get("VLLM_USE_V1", "1"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64"))

try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.layers import quantization as qmod
    from vllm.model_executor.models.registry import ModelRegistry
except Exception as e:
    print("Import vllm failed:", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(1)

# importa e registra la classe custom del progetto
try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
except Exception as e:
    print("Import/register deepseek_ocr failed:", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(1)

print("quant module:", qmod.__file__)
print("Available methods:", getattr(qmod, "QUANTIZATION_METHODS", None))
print("get_quantization_config signature:", inspect.signature(qmod.get_quantization_config))

method = "awq"
print(f"\nUsing method: {method}")
try:
    qc_cls = qmod.get_quantization_config(method)
    print("qc class/type:", qc_cls)
    try:
        print("qc class signature:", inspect.signature(qc_cls))
    except Exception:
        pass
except Exception as e:
    print("get_quantization_config failed:", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(2)

# AWQ supports 4-bit only -> try weight_bits=4
weight_bits = 4

# sensible combos to try
group_sizes = [128, 64, 256]
zero_point_opts = [True, False]
modules_exclude_list = [
    None,
    ["vision_model", "sam_model", "projector"],   # don't quantize vision parts (safer)
    ["sam_model", "vision_model"],                 # variant
]

# engine resource combos to attempt if instantiation succeeds but creation OOMs
engine_resource_attempts = [
    {"gpu_memory_utilization": 0.2, "swap_space": 8},
    {"gpu_memory_utilization": 0.15, "swap_space": 12},
    {"gpu_memory_utilization": 0.1, "swap_space": 12},
]

success = False
for gs in group_sizes:
    for zp in zero_point_opts:
        for mods in modules_exclude_list:
            print("\n--- Trying AWQ config:", {"weight_bits": weight_bits, "group_size": gs, "zero_point": zp, "modules_to_not_convert": mods})
            try:
                qc = qc_cls(weight_bits=weight_bits, group_size=gs, zero_point=zp, modules_to_not_convert=mods)
                print("QuantizationConfig instantiated:", type(qc))
                print("repr(qc):", repr(qc)[:400])
            except Exception as e:
                print("Failed to instantiate QuantizationConfig:", type(e).__name__, e)
                traceback.print_exc(limit=4)
                continue

            # try creating engine with resource attempts
            for res in engine_resource_attempts:
                print(" -> Trying engine create with resources:", res)
                args = AsyncEngineArgs(
                    model="deepseek-ai/DeepSeek-OCR",
                    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                    block_size=256,
                    max_model_len=8192,
                    enforce_eager=False,
                    trust_remote_code=True,
                    skip_tokenizer_init=True,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=res["gpu_memory_utilization"],
                    swap_space=res["swap_space"],
                    quantization=qc,
                )
                try:
                    t0 = time.time()
                    engine = AsyncLLMEngine.from_engine_args(args)
                    elapsed = time.time() - t0
                    print(f"Engine created in {elapsed:.1f}s with config {res}. Now shutting down...")
                    try:
                        engine.shutdown()
                    except Exception as e:
                        print("Shutdown raised:", type(e).__name__, e)
                    print("Success! Engine created+shutdown OK with AWQ config:", {"weight_bits": weight_bits, "group_size": gs, "zero_point": zp, "modules_to_not_convert": mods, "engine_res": res})
                    success = True
                    break
                except Exception as e:
                    print("Engine creation FAILED:", type(e).__name__, e)
                    traceback.print_exc(limit=6)
                    # try next resource combo
            if success:
                break
        if success:
            break
    if success:
        break

if not success:
    print("\nNo combination succeeded. Next options:")
    print(" - Try keeping more modules unquantized (e.g., exclude vision parts).")
    print(" - Try transformers+bitsandbytes 8-bit fallback (I can prepare script).")
    print(" - Try CPU-only or use a machine with more VRAM.")
else:
    print("\nAWQ quantization succeeded with one of the tested combos. Use the printed config for real runs.")

sys.exit(0 if success else 2)