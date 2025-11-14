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
# Use V1 engine off for consistency with other launchers on this project
os.environ['VLLM_USE_V1'] = os.environ.get('VLLM_USE_V1', '0')
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
    gpu_memory_utilization=0.2,     # conservative default to avoid OOM on small GPUs
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
# -----------------------------
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def clean_formula(text: str) -> str:
    formula_pattern = r'\\\[(.*?)\\\]'

    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'

    return re.sub(formula_pattern, process_formula, text)


def re_match(text: str):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_other = [a_match[0] for a_match in matches]
    return matches, mathes_other


def process_single_image(image: Image.Image):
    """Prepare the LLM request item for a single image."""
    prompt_in = prompt
    return {
        "prompt": prompt_in,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)
        },
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print(f'{Colors.RED}glob images.....{Colors.RESET}')

    # Collect image paths (support many extensions)
    images_path = sorted([p for p in glob.glob(os.path.join(INPUT_PATH, '*')) if p.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not images_path:
        print(f"No images found in {INPUT_PATH}", file=sys.stderr)
        raise SystemExit(1)

    # Load images safely
    images = []
    for image_path in images_path:
        try:
            img = Image.open(image_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: could not open {image_path}: {e}")

    prompt = PROMPT

    # Preprocess images in parallel (thread pool)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processed images"
        ))

    # Generate outputs (LLM handles batching/permutations)
    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )

    output_path = OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)

    for output, image_path in zip(outputs_list, images_path):
        try:
            content = output.outputs[0].text
        except Exception as e:
            print(f"Warning: empty/invalid output for {image_path}: {e}")
            continue

        # Save raw content
        mmd_det_path = os.path.join(output_path, os.path.basename(image_path).replace('.jpg', '_det.md').replace('.jpeg', '_det.md').replace('.png', '_det.md'))
        with open(mmd_det_path, 'w', encoding='utf-8') as afile:
            afile.write(content)

        # Clean and postprocess
        content = clean_formula(content)
        matches_ref, mathes_other = re_match(content)
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')

        mmd_path = os.path.join(output_path, os.path.basename(image_path).replace('.jpg', '.md').replace('.jpeg', '.md').replace('.png', '.md'))
        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(content)

    print("Batch processing complete.")
