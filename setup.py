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
        "diffusers>=0.31.0",
        "transformers>=4.49.0",
        "safetensors",
        "einops",
        "loguru",
        "tqdm",
        "huggingface_hub",
    ],
    extras_require={
        "train": [
            "peft>=0.10.0",
            "accelerate>=1.1.1",
        ],
        "dev": [
            "pytest",
            "ruff",
        ],
    },
)
