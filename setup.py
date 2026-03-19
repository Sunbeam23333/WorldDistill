from setuptools import setup, find_packages

setup(
    name="worlddistill",
    version="0.1.0",
    description="Video Generation & World Model Distillation Toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/worlddistill/WorldDistill",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "diffusers>=0.33.0,<1",
        "transformers==4.57.1",
        "tokenizers>=0.22.0,<0.23",
        "safetensors>=0.4.5,<1",
        "einops",
        "loguru",
        "tqdm",
        "huggingface_hub>=0.25.0,<1",
        "PyYAML>=6.0,<7",
        "packaging>=24.2,<26",
    ],
    extras_require={
        "train": [
            "peft>=0.10.0",
            "accelerate>=1.1.1",
            "tensorboard>=2.16.2",
            "wandb>=0.19.0",
        ],
        "dev": [
            "pytest",
            "ruff",
        ],
    },
)
