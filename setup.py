#!/usr/bin/env python
"""
dLLM Reasoning: 迭代精炼训练框架
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dllm-reasoning",
    version="1.1.0",
    author="zihan liu",
    description="迭代精炼训练框架，用于训练扩散语言模型进行推理任务",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "transformers>=4.57.1",
        "hydra-core>=1.3.2",
        "tensordict>=0.10.0",
        "pandas>=2.0.0",
        "pyarrow>=20.0.0",
        "wandb>=0.20.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
