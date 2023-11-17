from typing import List
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
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

    prompt_token_ids = sample_token_sequences(vocab_size, args.batch_size, args.input_len, 0)

    def run_to_completion(profile:bool=False):
        start_time = time.perf_counter()
        llm.generate(prompt_token_ids=prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    print("Warming up...")
    run_to_completion(profile=False)

    latencies = []

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()

    for i in tqdm(range(args.num_iters), desc="Profiling iterations"):
        if args.profile:
            torch.cuda.nvtx.range_push(f"iteration{i}")
        latencies.append(run_to_completion(profile=args.profile))
        if args.profile:
            torch.cuda.nvtx.range_pop()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    mean_latency = np.mean(latencies)
    print(f'Avg latency: {mean_latency} seconds')

    num_tokens = args.batch_size * (args.input_len + args.output_len)
    print(f'Number of tokens: {num_tokens}')
    print(f'{num_tokens / mean_latency} tokens/s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile processing a single batch of requests using Nsight.')
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
    parser.add_argument('--profile',
                        action='store_true',
                        help='profile CUDA code')
    
    args = parser.parse_args()
    main(args)
