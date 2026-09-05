from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="worlddistill",
    version="0.2.0",
    description="Training and runtime framework for video generators and world models",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Sunbeam23333/WorldDistill",
    license="Apache-2.0",
    license_files=[
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES",
        "third_party/licenses/TENCENT_HY_WORLDPLAY_COMMUNITY_LICENSE.txt",
    ],
    packages=find_packages(exclude=["tests", "docs"]),
    py_modules=["distill_capabilities", "cuda_compat"],
    python_requires=">=3.10,<3.13",
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
        "packaging>=24.2,<27",
        "numpy",
    ],
    extras_require={
        "train": [
            "peft>=0.17.0,<1",
            "accelerate>=0.34.2,<2",
            "torchvision>=0.20.1",
            "av",
            "decord",
            "tensorboard>=2.16.2",
            "wandb>=0.19.0",
        ],
        "inference": [
            "accelerate>=0.34.2,<2",
            "torchvision>=0.20.1",
            "torchaudio>=2.5.1",
            "scipy",
            "opencv-python>=4.9.0.80",
            "imageio[ffmpeg]",
            "imageio-ffmpeg",
            "av",
            "decord",
            "Pillow",
            "qtorch",
            "easydict",
            "ftfy",
            "gguf",
            "prometheus-client",
            "pydantic>=2,<3",
            "requests>=2.31,<3",
        ],
        "cuda-kernels": [
            "flash-attn>=2.5.0",
            "sgl-kernel",
        ],
        "distributed": ["deepspeed>=0.14.0"],
        "dev": [
            "pytest",
            "ruff",
        ],
    },
    project_urls={
        "Source": "https://github.com/Sunbeam23333/WorldDistill",
        "Issues": "https://github.com/Sunbeam23333/WorldDistill/issues",
    },
)
