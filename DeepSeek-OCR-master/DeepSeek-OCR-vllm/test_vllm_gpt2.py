# /tmp/test_vllm_gpt2.py
import os, sys
# imposta env prima degli import critici
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_USE_V1'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
# opzionale per debug
os.environ['VLLM_LOG_LEVEL'] = 'debug'

from vllm import LLM, SamplingParams
import time, traceback

def main():
    try:
        print("Creating small LLM (gpt2)...", flush=True)
        llm = LLM(
            model="gpt2",
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.1,
            tensor_parallel_size=1,
            max_model_len=512,
        )
        print("LLM created. Generating...", flush=True)
        sampling = SamplingParams(temperature=0.0, max_tokens=20)
        prompts = [{"prompt": "Hello world, my name is"}]
        outputs = llm.generate(prompts, sampling_params=sampling)
        print("Got outputs:", flush=True)
        for out in outputs:
            try:
                print(">>>", out.outputs[0].text)
            except Exception as e:
                print("No text in output:", e)
        print("Shutting down LLM...", flush=True)
        llm.shutdown()
        print("Done.", flush=True)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    main()
