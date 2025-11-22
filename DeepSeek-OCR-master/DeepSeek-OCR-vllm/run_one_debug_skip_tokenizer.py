#!/usr/bin/env python3
import os, sys, asyncio

# MUST set envs before importing vllm/transformers/torch
os.environ['VLLM_USE_V1'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Optional extra debug
os.environ['VLLM_LOG_LEVEL'] = 'debug'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

from PIL import Image, ImageOps
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, PROMPT

def make_image_feat(img_path):
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert('RGB')
    feat = DeepseekOCRProcessor().tokenize_with_images(images=[img], bos=True, eos=True, cropping=True)
    return feat

def make_engine():
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        skip_tokenizer_init=True,   # IMPORTANT: avoid tokenizer init inside engine
    )
    print("Creating AsyncLLMEngine ...", flush=True)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Engine created.", flush=True)
    return engine

async def run_once(image_feat, engine):
    sampling = SamplingParams(temperature=0.0, max_tokens=256, top_k=1)
    req = {"prompt": PROMPT, "multi_modal_data": {"image": image_feat}}
    printed = 0
    print("Calling engine.generate ...", flush=True)
    try:
        async for out in engine.generate(req, sampling, "debug-one"):
            if out.outputs:
                full = out.outputs[0].text
                new = full[printed:]
                if new:
                    sys.stdout.write(new); sys.stdout.flush()
                printed = len(full)
    except Exception as e:
        print("Exception during generate:", e, flush=True)
    finally:
        try:
            await engine.shutdown()
        except Exception as e:
            print("engine.shutdown error:", e, flush=True)
    print("\nDONE", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python /tmp/run_one_debug_skip_tokenizer.py /path/to/image.png")
        sys.exit(1)
    img_path = sys.argv[1]
    print("Image:", img_path, flush=True)
    image_feat = make_image_feat(img_path)
    engine = make_engine()
    asyncio.run(run_once(image_feat, engine))