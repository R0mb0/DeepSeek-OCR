#!/usr/bin/env python3
"""
Esegui un test "one-shot" creando un AsyncLLMEngine quantizzato (bits=8) usando
la factory di vllm.get_quantization_config quando disponibile.

Usage:
  conda activate deepseek-ocr
  export TOKENIZERS_PARALLELISM=false
  export VLLM_USE_V1=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
  python run_quantized_debug.py /percorso/alla/immagine.png

Il file tenta automaticamente una lista di metodi di quantizzazione tramite
vllm.model_executor.layers.quantization.get_quantization_config.
Se trova una config utilizzabile la usa per creare l'engine (create+shutdown),
poi tenta una singola generazione streaming per l'immagine fornita.
"""

import os
import sys
import time
import traceback

# --- Impostazioni ambiente prima di importare vllm/transformers/torch ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64"))

# -------------------------
# Imports (dopo env)
# -------------------------
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.models.registry import ModelRegistry
    import vllm.model_executor.layers.quantization as qmod
except Exception as e:
    print("Errore import vllm o moduli correlati:", type(e).__name__, e)
    traceback.print_exc()
    sys.exit(1)

# import local model + processor
try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    from process.image_process import DeepseekOCRProcessor
    from config import MODEL_PATH, PROMPT
except Exception as e:
    print("Errore import progetto locale (deepseek_ocr/process/config):", type(e).__name__, e)
    traceback.print_exc()
    sys.exit(1)

# register model class for vllm
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def find_quant_config(bits: int = 8):
    """
    Prova a costruire una QuantizationConfig concreta usando la factory get_quantization_config.
    Restituisce (method_name, qc_object) oppure (None, None) se non trova nulla.
    """
    print("quant module:", qmod.__file__)
    methods = []
    qm = getattr(qmod, "QUANTIZATION_METHODS", None)
    if isinstance(qm, dict):
        methods = list(qm.keys())
    elif isinstance(qm, (list, tuple, set)):
        methods = list(qm)
    # aggiungi nomi comuni se non presenti
    extra = ["bitsandbytes", "bnb", "bnb_linear", "bnb_4bit", "gptq", "gptq_cuda", "llm.int8", "int8"]
    for n in extra:
        if n not in methods:
            methods.append(n)

    print("Trying quantization methods:", methods)
    for m in methods:
        try:
            print(f"  -> trying get_quantization_config('{m}', bits={bits}) ...", end=" ", flush=True)
            qc = qmod.get_quantization_config(m, bits=bits)
            print("OK")
            return m, qc
        except Exception as e:
            print("failed:", type(e).__name__, str(e))
    return None, None


def make_image_feature(img_path: str):
    """Crea la feature immagine (tokenized) usando DeepseekOCRProcessor."""
    from PIL import Image, ImageOps
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    proc = DeepseekOCRProcessor()
    feat = proc.tokenize_with_images(images=[img], bos=True, eos=True, cropping=True)
    return feat


def run_test(image_path: str, bits: int = 8, gpu_util: float = 0.2, swap_space_gb: int = 8):
    method, qc = find_quant_config(bits=bits)
    if qc is None:
        print("Nessuna QuantizationConfig trovata automaticamente. Interrompo.")
        return 2

    print(f"Usando metodo di quantizzazione: {method}, object: {type(qc)}")

    # create engine args
    args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_util,
        swap_space=swap_space_gb,
        quantization=qc,
    )

    print("Creating AsyncLLMEngine (quantized)...")
    t0 = time.time()
    try:
        engine = AsyncLLMEngine.from_engine_args(args)
    except Exception as e:
        print("Errore durante la creazione dell'engine:", type(e).__name__, e)
        traceback.print_exc()
        return 3
    print("Engine creato in %.1f s" % (time.time() - t0))

    # prepare image feature
    try:
        image_feat = make_image_feature(image_path)
    except Exception as e:
        print("Errore creando image feature:", e)
        traceback.print_exc()
        try:
            engine.shutdown()
        except Exception:
            pass
        return 4

    sampling = SamplingParams(temperature=0.0, max_tokens=256, top_k=1)
    req = {"prompt": PROMPT, "multi_modal_data": {"image": image_feat}}
    printed = 0
    print("Calling engine.generate ... (streaming output)\n")
    try:
        # engine.generate is async generator; use .generate(...) in sync context via asyncio loop
        import asyncio

        async def _run():
            nonlocal printed
            async for out in engine.generate(req, sampling, "debug-quant"):
                if out.outputs:
                    full = out.outputs[0].text
                    new = full[printed:]
                    if new:
                        sys.stdout.write(new)
                        sys.stdout.flush()
                    printed = len(full)
        asyncio.run(_run())
    except Exception as e:
        print("\nException during generate:", type(e).__name__, e)
        traceback.print_exc()
    finally:
        try:
            print("\nShutting down engine ...")
            engine.shutdown()
            print("Shutdown OK")
        except Exception as e:
            print("Errore in shutdown:", type(e).__name__, e)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_quantized_debug.py /percorso/alla/immagine.png")
        sys.exit(1)
    img = sys.argv[1]
    rc = run_test(img, bits=8, gpu_util=0.2, swap_space_gb=8)
    sys.exit(rc)