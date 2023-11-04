# Ray vLLM Interence

A simple simple service that integrates [vLLM](https://github.com/vllm-project/vllm) with [Ray Serve](https://github.com/ray-project/ray) for fast and scalable LLM serving.

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

 Install:

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

### Use AWQ Quantization

[Activation-aware Weight Quantization (AWQ)](https://github.com/mit-han-lab/llm-awq) is an 4-bit quantization method for LLMs.

Launch the service with a quantized Llama-2-7b model:

    serve run ray_vllm_inference.vllm_serve:deployment model="asprenger/meta-llama-Llama-2-7b-chat-hf-gemm-w4-g128-awq" quantization="awq"

Call the service with a system prompt and a user message:

    curl --header "Content-Type: application/json" --data '{ "messages":[{"role":"system", "content":"You are an expert assistant. Always give a short reply."}, {"role":"user", "content":"What is the capital of France?"}], "max_tokens":32, "temperature":0}' http://127.0.0.1:8000/generate

### Using SqueezeLLM

[SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM)

serve run ray_vllm_inference.vllm_serve:deployment model="squeeze-ai-lab/sq-llama-2-7b-w4-s0" quantization="squeezellm"

curl --header "Content-Type: application/json" --data '{ "prompt":"The capital of France is ","max_tokens":32, "temperature":0}' http://127.0.0.1:8000/generate


### Streaming reponse

Test streaming response:

    python -m ray_vllm_inference.client --stream --max-tokens 512 --user-message "What can I do on a weekend trip to London?"

### Deploy on a Ray cluster

You can deploy the service on a running Ray cluster using the [serve deploy CLI](https://docs.ray.io/en/latest/serve/api/index.html#command-line-interface-cli) command and a [serve config](https://docs.ray.io/en/latest/serve/production-guide/config.html):

    serve run deploy-config/vllm_serve_config.yaml

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

Benchmark Llama-2-7b

    python benchmark_throughput.py --model="meta-llama/Llama-2-7b-chat-hf"

Benchmark Llama-2-7b with AWQ:

    python benchmark_throughput.py --model="asprenger/meta-llama-Llama-2-7b-chat-hf-gemm-w4-g128-awq" --quantization="awq" 

Benchmark Llama-2-7b with SqueezeLLM:

    python benchmark_throughput.py --model="squeeze-ai-lab/sq-llama-2-7b-w4-s0" --quantization="squeezellm"