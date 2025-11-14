#!/usr/bin/env python3
"""
Patched launcher per DeepSeek-OCR (evita deadlock fork+threads, crea engine correttamente,
tokenizza DOPO la creazione dell'engine e gestisce directory/file di input).
Salva questo file al posto del launcher corrente oppure usalo come riferimento.
"""

# -------------------------
# ENV + multiprocessing FIX
# -------------------------
# IMPORTANT: environment variables and multiprocessing start method MUST be set
# before importing libraries that spawn threads (tokenizers, torch, vllm, ...).
import os
import multiprocessing

# Coerenza V1 engine / vllm
os.environ['VLLM_USE_V1'] = os.environ.get('VLLM_USE_V1', '1')
os.environ['TOKENIZERS_PARALLELISM'] = os.environ.get('TOKENIZERS_PARALLELISM', 'false')
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# Limit native thread pools to avoid extra threads at fork time
os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '1')
os.environ['MKL_NUM_THREADS'] = os.environ.get('MKL_NUM_THREADS', '1')

# Force spawn start method to avoid fork() with threads deadlocks
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # already set - ignore
    pass

# -------------------------
# Imports (after env set)
# -------------------------
import asyncio
import re
import time
import sys
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

import torch

# vllm / model imports
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

# local model and processing
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# import config AFTER we set envs above (config may instantiate tokenizer)
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE

# register model class
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# -------------------------
# Utility functions
# -------------------------
def load_image(image_path):
    try:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        return image
    except Exception as e:
        print(f"error loading image {image_path}: {e}", file=sys.stderr)
        try:
            return Image.open(image_path)
        except Exception:
            return None


def ensure_image_list(input_path):
    """Return a list of image file paths (handles directory or single file)."""
    p = Path(input_path)
    if p.is_dir():
        imgs = sorted(
            [p / f for f in sorted(os.listdir(p)) if str(f).lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        if not imgs:
            raise FileNotFoundError(f"No image files found in directory {input_path}")
        return [str(x) for x in imgs]
    elif p.is_file():
        return [str(p)]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print("extract_coordinates_and_label error:", e, file=sys.stderr)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")
                        except Exception as e:
                            print("save crop error:", e, file=sys.stderr)
                            pass
                        img_idx += 1
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                       fill=(255, 255, 255, 30))
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception:
                        pass
        except Exception:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


# -------------------------
# Async generation helpers
# -------------------------
async def generate_for_image(engine, pil_image, prompt, request_id=None):
    """
    Tokenize the PIL image AFTER the engine is created and then stream generate.
    Returns full text result.
    """
    # Tokenize (synchronous call) - safe after engine created
    if '<image>' in prompt:
        processor = DeepseekOCRProcessor()
        image_feat = processor.tokenize_with_images(images=[pil_image], bos=True, eos=True, cropping=CROP_MODE)
    else:
        image_feat = None

    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    if request_id is None:
        request_id = f"request-{int(time.time())}"

    printed_length = 0
    final_output = ""

    if image_feat is not None:
        request = {"prompt": prompt, "multi_modal_data": {"image": image_feat}}
    else:
        request = {"prompt": prompt}

    print(f"DEBUG: calling engine.generate for {request_id}", flush=True)
    try:
        async for out in engine.generate(request, sampling_params, request_id):
            if out.outputs:
                full_text = out.outputs[0].text
                new_text = full_text[printed_length:]
                if new_text:
                    sys.stdout.write(new_text)
                    sys.stdout.flush()
                printed_length = len(full_text)
                final_output = full_text
    except Exception as e:
        print("Exception during engine.generate:", e, file=sys.stderr)
    print("\nDEBUG: finished generate for", request_id, flush=True)
    return final_output


# -------------------------
# Main async flow
# -------------------------
async def main_async(image_paths):
    # Create engine args - skip tokenizer init inside core to avoid double init issues
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        # Conservative GPU settings to reduce OOM on small GPUs:
        gpu_memory_utilization=0.2,  # reduce VRAM footprint on GPU
        swap_space=8,                # 8 GiB swap to host (vllm expects GiB)
        skip_tokenizer_init=True,    # important to avoid tokenizer init inside engine core
    )

    print("DEBUG: Creating AsyncLLMEngine ...", flush=True)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("DEBUG: AsyncLLMEngine created.", flush=True)

    # Ensure OUTPUT dirs
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_PATH}/images").mkdir(parents=True, exist_ok=True)

    # Iterate images sequentially (can be adapted to batch if needed)
    for img_path in image_paths:
        print(f"\nProcessing {img_path} ...", flush=True)
        pil_img = load_image(img_path)
        if pil_img is None:
            print(f"Unable to open {img_path}, skipping.", file=sys.stderr)
            continue
        pil_img = pil_img.convert("RGB")

        # generate and stream
        full_out = await generate_for_image(engine, pil_img, PROMPT, request_id=f"req-{Path(img_path).name}")

        # save results
        if full_out and '<image>' in PROMPT:
            print('=' * 10 + f' Saving result for {Path(img_path).name} ' + '=' * 10, flush=True)
            # original markdown
            result_ori_path = f'{OUTPUT_PATH}/result_ori_{Path(img_path).name}.mmd'
            with open(result_ori_path, 'w', encoding='utf-8') as af:
                af.write(full_out)

            matches_ref, matches_images, mathes_other = re_match(full_out)
            result_img = draw_bounding_boxes(pil_img.copy(), matches_ref)

            # replace image refs
            for m_idx, a_match_image in enumerate(matches_images):
                full_out = full_out.replace(a_match_image, f'![](images/{m_idx}.jpg)\n')

            # clean other refs
            for o_idx, a_match_other in enumerate(mathes_other):
                full_out = full_out.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
            result_path = f'{OUTPUT_PATH}/result_{Path(img_path).name}.mmd'
            with open(result_path, 'w', encoding='utf-8') as af:
                af.write(full_out)

            # save annotated image
            boxes_path = f'{OUTPUT_PATH}/result_with_boxes_{Path(img_path).name}.jpg'
            result_img.save(boxes_path)

    # Shutdown engine
    try:
        print("DEBUG: Shutting down engine ...", flush=True)
        await engine.shutdown()
        print("DEBUG: Engine shutdown complete.", flush=True)
    except Exception as e:
        print("Error shutting down engine:", e, file=sys.stderr)


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    try:
        image_list = ensure_image_list(INPUT_PATH)
    except Exception as e:
        print("INPUT_PATH error:", e, file=sys.stderr)
        sys.exit(1)

    # Run the async main
    asyncio.run(main_async(image_list))
