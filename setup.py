from typing import List
import os
from pathlib import Path
from setuptools import setup, find_packages

def read_readme() -> str:
    """Read the README file."""
    return (Path(__file__).parent / "README.md").read_text(encoding="UTF-8")

common_setup_kwargs = {
    "name": "ray_vllm_inference",
    "version": "0.1.0",
    "description": "An example how to integrate vLLM with Anyscale Ray.",
    "author": "Andre Sprenger",
    "license": "Apache 2.0",
    "license_files": "LICENSE.txt",
    "python_requires": ">=3.8.0",
    "long_description": read_readme(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/asprenger/ray_vllm_inference",
    "keywords": ["python", "vllm", "ray", "llm"],
    "platforms": ["linux"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
}

requirements = [
    "ray==2.7.1",
    "ray[serve]==2.7.1",
    "pydantic==1.10.13", # fix problem with Ray Serve startup
    #"vllm==0.2.1.post1", # 0.2.1.post1 is a fix for broken version 0.2.1 (Bin tensor parallelism support)
    "vllm @ git+https://github.com/vllm-project/vllm.git@aa9af07cac7b681d7195dbd5de621fc5a2acde99",
    "protobuf==3.20.3"
]



setup(
    packages=find_packages(),
    install_requires=requirements,
    **common_setup_kwargs
)