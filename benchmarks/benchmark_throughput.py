from typing import List
import argparse
import numpy as np
import random
import time
from vllm import LLM, SamplingParams

# Assume that the token input/output length is normally distributed.
# The distribution params are from Anyscale's "Reproducible Performance Metrics" 
# blog post.
PROMPT_LENGTH_MEAN = 550
PROMPT_LENGTH_STD = 150
OUTPUT_LENGTH_MEAN = 150
OUTPUT_LENGTH_STD = 20

def sample_token_sequences(vocab_size:int, num_seq: int, len_mean:int, len_std:int) -> List[List[int]]:
    return [np.random.randint(0, high=vocab_size, size=int(np.random.normal(loc=len_mean, scale=len_std))).tolist() 
            for i in range(num_seq)]

def run_benchmark(llm:LLM, requests: List[List[int]], use_tqdm:bool) -> float:

    # Add the requests to the engine. Internally this appends the requests to
    # the vLLM Scheduler waiting queue but does not start the processing.
    print(f"Add {len(requests)} requests to vLLM")
    total_output_tokens = 0
    for prompt_token_ids in requests:
        max_tokens = int(np.random.normal(loc=OUTPUT_LENGTH_MEAN, scale=OUTPUT_LENGTH_STD))
        total_output_tokens += max_tokens
        sampling_params = SamplingParams(
            temperature=1.0,
            ignore_eos=True,
            max_tokens=max_tokens
        )
        llm._add_request(
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params
        )

    total_input_tokens = sum(len(req) for req in requests)

    # Start generating output sequences until all requests have
    # been processed.
    print(f"Start output generation")
    start = time.perf_counter()
    llm._run_engine(use_tqdm=use_tqdm)
    end = time.perf_counter()
    return end - start, total_input_tokens, total_output_tokens

def main(args: argparse.Namespace):
    print(args)
    np.random.seed(args.seed)
    num_prompts = args.num_prompts

    llm = LLM(
        model=args.model,
        download_dir=args.download_dir,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype
    )

    vocab_size = llm.llm_engine.tokenizer.vocab_size
    requests = sample_token_sequences(vocab_size, num_prompts, PROMPT_LENGTH_MEAN, PROMPT_LENGTH_STD)
    elapsed_time, total_input_tokens, total_output_tokens = run_benchmark(llm, requests, args.use_tqdm)
    total_tokens = total_input_tokens + total_output_tokens

    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Requests: {len(requests)}")
    print(f"Input tokens: {total_input_tokens}, output tokens: {total_output_tokens} ")
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_tokens / elapsed_time:.2f} tokens/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="the model ID")
    parser.add_argument('--quantization',
                        choices=['awq', 'squeezellm', None],
                        default=None,
                        help="the quantization method")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--num-prompts",
                        type=int,
                        default=500,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--use-tqdm',
                        action='store_true',
                        help='show progress bar')
    parser.add_argument('--dtype',
                        type=str,
                        default='auto',
                        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
                        help='data type for model weights and activations. '
                        'The "auto" option will use FP16 precision for FP32 and FP16 '
                        'models, and BF16 precision for BF16 models.')
    args = parser.parse_args()
    main(args)