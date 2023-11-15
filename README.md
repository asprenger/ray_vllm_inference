# Ray vLLM Interence

A service that integrates [vLLM](https://github.com/vllm-project/vllm) with [Ray Serve](https://github.com/ray-project/ray) for fast and scalable LLM serving.

vLLM is an open source LLM inference engine that supports the following features:

  * Efficient KV cache memory management with PagedAttention
  * AWQ quantization
  * Continuous batching
  * Streaming output
  * Efficient implementation of decoding strategies (parallel decoding, beam search, etc.)
  * Multi-GPU support
  * Integration with HuggingFace

Deploying vLLM instances with Ray Serve provides the additional features:

  * Multi-server model deployment
  * Autoscaling
  * Failure recovery

## Setup

Requirements:

 * OS: Linux
 * Python: 3.8 or higher
 * GPU: CUDA compute capability 7.0 or higher (V100, T4, A2, A16, A10, A100, H100, etc.)
 * CUDA Toolkit 11.8 and later

Install from Github:

    pip install git+https://github.com/asprenger/ray_vllm_inference

Install in develop mode:

    git clone https://github.com/asprenger/ray_vllm_inference
    cd ray_vllm_inference
    pip install -e .

## Usage

### Getting started

Lauch the service with a `facebook/opt-125m` model from HuggingFace:

    serve run ray_vllm_inference.vllm_serve:deployment model="facebook/opt-125m"

This command launches a local Ray cluster, downloads the model from HuggingFace and starts a Ray Serve instance on localhost port 8000.

Call the service with a simple prompt:

    curl --header "Content-Type: application/json" --data '{ "prompt":"The capital of France is ", "max_tokens":32, "temperature":0}' http://127.0.0.1:8000/generate

Note that `facebook/opt-125m` is a toy model and the output is often garbled.

See `ray_vllm_inference/protocol.py::GenerateRequest` for a list of supported request parameters.

### Use a Llama-2 model

The official Llama-2 models on HuggingFace are [gated models](https://huggingface.co/docs/hub/models-gated) that require 
access permission. To use the model you need a [HuggingFace access token](https://huggingface.co/docs/hub/security-tokens) 
with READ permission.

    export HUGGING_FACE_HUB_TOKEN={YOUR_HF_TOKEN}
    serve run ray_vllm_inference.vllm_serve:deployment model="meta-llama/Llama-2-7b-chat-hf"

This command launches a local Ray cluster and starts a Ray Serve instance that listens on localhost port 8000.

Call the service with a system prompt and a user message:

    curl --header "Content-Type: application/json" --data '{ "messages":[{"role":"system", "content":"You are an expert assistant. Always give a short reply."}, {"role":"user", "content":"What is the capital of France?"}], "max_tokens":32, "temperature":0}' http://127.0.0.1:8000/generate

### Use a Llama-2 model with AWQ quantization

[Activation-aware Weight Quantization (AWQ)](https://github.com/mit-han-lab/llm-awq) is an 4-bit quantization method for LLMs.

Launch the service with a quantized Llama-2-7b model:

    serve run ray_vllm_inference.vllm_serve:deployment model="asprenger/meta-llama-Llama-2-7b-chat-hf-gemm-w4-g128-awq" quantization="awq"

Call the service with a system prompt and a user message:

    curl --header "Content-Type: application/json" --data '{ "messages":[{"role":"system", "content":"You are an expert assistant. Always give a short reply."}, {"role":"user", "content":"What is the capital of France?"}], "max_tokens":32, "temperature":0}' http://127.0.0.1:8000/generate

### Streaming reponse

Test streaming response:

    python -m ray_vllm_inference.streaming_client --max-tokens 2048 --user-message "What can I do on a weekend trip to London?"

## Benchmarks

## HTTP service

[Apache Benchmark](https://httpd.apache.org/docs/2.4/programs/ab.html) is a simple tool to benchmark HTTP services.

Install Apache Benchmark:

    sudo apt update
    sudo apt -y install apache2-utils

Create a file `postdata.json` with a POST request payload. For example:

    {"prompt":"TEST_PROMPT with length N tokens", "max_tokens":128, "temperature":0, "ignore_eos":true}

When benchmarking a LLM you usually want to fix the length of the input prompt and the length of the
generated output. The `ignore_eos` flag forces the LLM to always generate `max_tokens`. 

Run a benchmark with 1000 requests and 4 concurrent clients:

    ab -T "application/json" -n 1000 -c 4 -p postdata.json http://127.0.0.1:8000/generate

## vLLM throughput

Benchmark Llama-2-7b:

    python benchmark_throughput.py --model="meta-llama/Llama-2-7b-chat-hf" --num-prompts 1000

Output on A100:

    Total time: 131.23s
    Requests: 1000
    Input tokens: 557060, output tokens: 149589
    Throughput: 7.62 requests/s, 5384.72 tokens/s

Benchmark Llama-2-7b with AWQ:

    python benchmark_throughput.py --model="asprenger/meta-llama-Llama-2-7b-chat-hf-gemm-w4-g128-awq" --quantization="awq" --num-prompts 1000

Output on A100:

    Total time: 250.26s
    Requests: 1000
    Input tokens: 557060, output tokens: 149589
    Throughput: 4.00 requests/s, 2823.66 tokens/s