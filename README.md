# PL-MAI_VPS

## Setup

Recommend & tested python version is 12.x

!! Running in a virtual environment is highly recommended !!

Install vLLM to run this repo, follow the guide for either:

- [GPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) (CUDA, AMD ROCm or Intel XPU)
- or [CPU](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html#intelamd-x86) (Intel/AMD x86, ARM
  AArch64, Apple silicon, IBM Z S390X)

Install dependencies in `requirements.txt`

## Full setup for Linux and CUDA 12.8 GPU

Using [pyenv](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html#intelamd-x86) for version management,
pip for package management (vLLM recommends uv), and venv for the virtual environment

```bash
python -m venv .venv 
source .venv/bin/activate
pip install --upgrade pip

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## vLLM test

Running `python src/pl_mai_vps/setup/vllm_test.p` runs the Qwen2.5-VL-3B-Instruct model and starts a console chat
dialog.

It reserves realistic memory necessary for 50 image frames out of 480p videos and inputs two images of unicorns via
URLs.

If the setup is working, the model can correctly answer questions about the images and answer general questions.

Note: The model (7.1 GB) will be downloaded and stored in `HF_HOME` (or on Linux default `~/.cache/huggingface/hub`).
This can take some time, depending on download speed.

## Run baseline

First make sure to activate the virtual environment

```bash
source .venv/bin/activate
```

then use:

```bash
python run.py
```

This, by default, runs the baseline inference

For additional options see:

```bash
python run.py --help
```
