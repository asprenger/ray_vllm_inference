from typing import List
import os
import sys
import re
from pathlib import Path
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

def read_readme() -> str:
    """Read the README file."""
    return (Path(__file__).parent / "README.md").read_text(encoding="UTF-8")

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

def get_version() -> str:
    version = find_version(get_path("ray_vllm_inference", "__init__.py"))
    return version

assert sys.version_info.major == 3
assert sys.version_info.minor in [8,9,10,11,12]
cp_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

requirements = [
    "ray==2.8.0",
    "ray[serve]==2.8.0",
    "pydantic==1.10.13", # fix problem with Ray Serve startup
    # vLLM v0.2.2, CUDA 11 (see: https://vllm.readthedocs.io/en/latest/getting_started/installation.html#install-with-pip)
    f"vllm @ git+https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-{cp_version}-{cp_version}-manylinux1_x86_64.whl",
    "protobuf==3.20.3"
]

setup(
    name="ray_vllm_inference",
    version="0.1.0",
    description="A service that integrates vLLM with Ray Serve for fast and scalable LLM serving.",
    author="Andre Sprenger",
    license="Apache 2.0",
    license_files="LICENSE.txt",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/asprenger/ray_vllm_inference",
    keywords=["python", "vllm", "ray", "llm"],
    platforms= ["linux"],
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=find_packages(include=['ray_vllm_inference', 'ray_vllm_inference.*']),
    install_requires=requirements,
    include_package_data=True,
    package_data={"ray_vllm_inference": ["models/*"]},
    python_requires=">=3.8.0",
)