# LLM Train

A hands-on repository for pre-training GPT-2 (124M) from scratch, covering tokenization, model architecture, and multi-platform accelerator support (CUDA, Apple MPS, Intel XPU, NPU/DirectML).

## 📚 Contents

- [LLM Train](#llm-train)
  - [📝 Overview](#-overview)
  - [🚀 Getting Started](#-getting-started)
  - [🧠 GPT-2 Training](#-gpt-2-training)
  - [🔤 Tokenization](#-tokenization)
  - [📄 Papers](#-papers)
  - [📖 References](#-references)
  - [📄 License](#-license)

## 📝 Overview

This repo includes:

- [GPT-2 pre-training scripts](gpt-2/) with a numbered learning progression from loading a pretrained model to gradient accumulation
- [BPE tokenization](gpt-tokenizer/) study and experiments with the tiktoken library
- [Apple MLX](mlx/) framework experiments
- [Reading notes](papers/) on foundational LLM papers
- [Environment tools](envtools/) for cross-platform setup (pip, conda, WSL2)

### Repository Structure

```
llm-train/
├── gpt-2/              # GPT-2 pre-training scripts (numbered learning progression)
│   ├── cuda/           #   NVIDIA CUDA training scripts
│   ├── mps/            #   Apple Silicon (MPS) training scripts
│   ├── winx64xpu/      #   Intel Arc GPU (XPU) training scripts
│   └── winx64npu/      #   Intel NPU / DirectML training scripts
├── gpt-tokenizer/      # BPE tokenization study & experiments
│   └── mps/            #   Apple Silicon tokenizer scripts
├── mlx/                # Apple MLX framework experiments
├── papers/             # Reading notes on key papers
├── envtools/           # Environment creation scripts & guides
├── data/               # Datasets (e.g. FashionMNIST)
└── cache/              # Model cache (Phi-3-mini, etc.)
```

### Accelerator Support

| Accelerator              | Directory          | Description                              |
| ------------------------ | ------------------ | ---------------------------------------- |
| **NVIDIA CUDA**          | `gpt-2/cuda/`     | Full training pipeline on NVIDIA GPUs    |
| **Apple MPS**            | `gpt-2/mps/`      | Training on Apple Silicon (M1/M2/M3/M4)  |
| **Intel XPU**            | `gpt-2/winx64xpu/`| Intel Arc GPU training via IPEX          |
| **NPU / DirectML**       | `gpt-2/winx64npu/`| Intel NPU training via DirectML          |

## 🚀 Getting Started

### 1. Clone the Repository

```powershell
git clone https://github.com/yingding/llm-train.git
cd llm-train
```

### 2. Set Up Your Environment

Pick your platform and package manager, then follow the dedicated setup guide:

| Platform                  | Guide                                                        | Quick Start                                              |
| ------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| **Windows (pip)**         | [envtools/README_winpip.md](envtools/README_winpip.md)       | [gpt-2/SETUP_WINX64_Native.md](gpt-2/SETUP_WINX64_Native.md) |
| **Windows (conda)**       | [envtools/README_winconda.md](envtools/README_winconda.md)   | [installminiconda.md](installminiconda.md)               |
| **macOS (Apple Silicon)** | [envtools/README_mac.md](envtools/README_mac.md)             | [gpt-2/SETUP.md](gpt-2/SETUP.md)                        |
| **Windows (WSL2)**        | [gpt-2/SETUP_WINX64_WSL2.md](gpt-2/SETUP_WINX64_WSL2.md)   | —                                                        |

> **Tip:** Environment creation scripts are available at [`envtools/create_env.ps1`](envtools/create_env.ps1) (Windows) and [`envtools/create_env.sh`](envtools/create_env.sh) (macOS/Linux) for automated venv setup.

### 3. Prerequisites

- Python 3.13+
- One of: NVIDIA GPU (CUDA), Apple Silicon (MPS), Intel Arc GPU (XPU), or Intel NPU (DirectML)

## 🧠 GPT-2 Training

The `gpt-2/` directory contains a step-by-step learning progression for pre-training GPT-2 124M, following [Andrej Karpathy's "Let's reproduce GPT-2"](https://www.youtube.com/watch?v=l8pRSuU81PU) lecture:

| Step | Script Prefix | Topic                            |
| ---- | ------------- | -------------------------------- |
| 1    | `01_`         | Load & test the pretrained model |
| 2    | `02_`         | Text generation                  |
| 3    | `03_`         | Random model initialization      |
| 4    | `04_`         | Single-batch training            |
| 5    | `05_`         | Multi-batch training             |
| 6    | `06_`         | Weight sharing scheme            |
| 7    | `07_`         | Residual stream scaling          |
| 8    | `08_`         | Scaled-up training               |
| 9    | `09_`         | Flash Attention                  |
| 10   | `10_`         | Even number vocab optimization   |
| 11   | `11_`         | Hyperparameter tuning            |
| 12   | `12_`         | Weight decay regularization      |
| 14   | `14_`         | Gradient accumulation            |

Each step is a standalone script with variants for each accelerator backend (CUDA, MPS, XPU, NPU).

See [gpt-2/README.md](gpt-2/README.md) for the full learning roadmap, references, and dataset details.

## 🔤 Tokenization

The `gpt-tokenizer/` directory covers Byte Pair Encoding (BPE) and the tiktoken library.

| Resource | Description |
| -------- | ----------- |
| [gpt-tokenizer/README.md](gpt-tokenizer/README.md) | Introduction to BPE and tokenization concepts |
| [gpt-tokenizer/Learning.md](gpt-tokenizer/Learning.md) | In-depth learning notes with code examples |

## 📄 Papers

Reading notes on foundational papers:

| File | Paper |
| ---- | ----- |
| [Radford2019.md](papers/Radford2019.md) | Language Models are Unsupervised Multitask Learners (GPT-2) |
| [Bavarian2022.md](papers/Bavarian2022.md) | Training Verifiers to Solve Math Word Problems |
| [Touvron2023.md](papers/Touvron2023.md) | LLaMA: Open and Efficient Foundation Language Models |
| [Rai2025.md](papers/Rai2025.md) | Recent research notes |

## 📖 References

- [Glossary](GLOSSARY.md) — Key terms and concepts
- [tech.md](tech.md) — SSH key setup for GitHub

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
