#!/usr/bin/env python3
"""
Quick CPU inference test:
 - loads DeepseekOCRForCausalLM (CPU)
 - tokenizes a small prompt & single image via DeepseekOCRProcessor
 - computes inputs_embeds and runs forward() -> compute_logits()
Usage:
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES=""
  python scripts/quick_cpu_infer.py --model deepseek-ai/DeepSeek-OCR --image /path/to/example.jpg
"""
import argparse
from PIL import Image, ImageOps
import torch
import importlib.util, sys
from transformers import AutoTokenizer

def load_class_from_cache(prefer="deepseekocr"):
    import os
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    files = list(cache_dir.rglob("modeling_deepseekocr.py")) + list(cache_dir.rglob("modeling_deepseekv2.py"))
    if not files:
        raise RuntimeError("No modeling files found in cache.")
    files = sorted(files, key=lambda p: prefer not in p.name)
    for f in files:
        spec = importlib.util.spec_from_file_location(f"mod_{f.stem}", str(f))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for cand in ["DeepseekOCRForCausalLM", "DeepseekV2ForCausalLM", "DeepseekOCRModel", "DeepseekV2Model"]:
            if hasattr(mod, cand):
                return getattr(mod, cand)
    raise RuntimeError("Candidate class not found in modeling files.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=False)
    args = parser.parse_args()

    model_id = args.model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cls = load_class_from_cache()
    print("Found class:", cls.__name__)

    print("Loading model (device_map=cpu, low_cpu_mem_usage=True)...")
    model = cls.from_pretrained(model_id, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()

    # build input ids (use tokenizer to encode prompt with a single image token if present)
    processor = None
    try:
        from process.image_process import DeepseekOCRProcessor
        processor = DeepseekOCRProcessor(tokenizer=tokenizer)
    except Exception:
        # fallback: create minimal processor without tokenizer
        raise RuntimeError("DeepseekOCRProcessor import failed; ensure project files are available.")

    if args.image:
        img = Image.open(args.image)
        img = ImageOps.exif_transpose(img).convert('RGB')
        feat = processor.tokenize_with_images(images=[img], bos=True, eos=True, cropping=True)
    else:
        # create a tiny zero image to test
        img = Image.new("RGB", (min(640, 256), min(640, 256)), color=(255,255,255))
        feat = processor.tokenize_with_images(images=[img], bos=True, eos=True, cropping=True)

    # create input ids from processor (they return the [input_ids, pixel_values, ...])
    input_ids = feat[0][0]  # LongTensor
    # create inputs_embeds
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings(input_ids, multimodal_embeddings=model.get_multimodal_embeddings(pixel_values=feat[0][1], images_crop=feat[0][2], images_spatial_crop=feat[0][4]))
        positions = torch.arange(inputs_embeds.size(1)).unsqueeze(0)
        hidden = model.forward(input_ids=None, positions=positions, intermediate_tensors=None, inputs_embeds=inputs_embeds)
        logits = model.compute_logits(hidden, None)
    print("Logits shape:", logits.shape if logits is not None else None)
    print("Quick inference OK (on CPU).")

if __name__ == "__main__":
    main()