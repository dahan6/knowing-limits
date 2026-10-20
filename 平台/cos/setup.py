from setuptools import setup, find_packages

setup(
    name="cos-framework",
    version="0.1.0",
    description="Deep Basin Cognitive Operating System Framework",
    author="Deep Basin",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "jieba>=0.42.1",
        "datasketch>=1.5.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
        "bitsandbytes>=0.40.0",
        "datasets>=2.14.0",
        "accelerate>=0.20.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
