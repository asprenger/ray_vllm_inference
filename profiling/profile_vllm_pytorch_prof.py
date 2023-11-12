"""Benchmark the latency of processing a single batch of requests."""
from typing import List
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

from vllm import LLM, SamplingParams

def sample_token_sequences(vocab_size:int, num_seq: int, len_mean:int, len_std:int) -> List[List[int]]:
    return [np.random.randint(0, high=vocab_size, size=int(np.random.normal(loc=len_mean, scale=len_std))).tolist() 
            for i in range(num_seq)]

def main(args: argparse.Namespace):

    print(args)

    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        #max_num_seqs=args.batch_size,
        #max_num_batched_tokens=args.batch_size * args.input_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )

    vocab_size = llm.llm_engine.tokenizer.vocab_size

    sampling_params = SamplingParams(
        temperature=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)

    dummy_prompt_token_ids = sample_token_sequences(vocab_size, args.batch_size, args.input_len, 0)

    def run_to_completion():
        start_time = time.perf_counter()
        llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    print("Warming up...")
    run_to_completion()

    latencies = []

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):

            for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
                latencies.append(run_to_completion())

    mean_latency = np.mean(latencies)
    print(f'Avg latency: {mean_latency:.2f} seconds')

    num_tokens = args.batch_size * (args.input_len + args.output_len)
    print(f'Number of tokens: {num_tokens}')
    print(f'{num_tokens / mean_latency:.2f} tokens/s')

    print('Prepare CPU time report')
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print('Save Chrome trace')
    prof.export_chrome_trace("chrome_trace.json")

    print('Save CPU stack traces')
    prof.export_stacks("profiler_CPU_stacks.txt", "self_cpu_time_total")

    print('Save CUDA stack traces')
    prof.export_stacks("profiler_CUDA_stacks.txt", "self_cuda_time_total")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile processing a single batch of requests using PyTorch Profiler.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--quantization',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=550)
    parser.add_argument('--output-len', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--dtype',
                        type=str,
                        default='auto',
                        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
                        help='data type for model weights and activations. '
                        'The "auto" option will use FP16 precision '
                        'for FP32 and FP16 models, and BF16 precision '
                        'for BF16 models.')    
    args = parser.parse_args()
    main(args)
