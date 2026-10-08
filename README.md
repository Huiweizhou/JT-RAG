# JT-RAG: A Jointly Trained Temporal Retrieval-Augmented Generation Framework for Biomedical Hypothesis Generation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains the preprocessing and training implementation of **"JT-RAG: A Jointly Trained Temporal Retrieval-Augmented Generation Framework for Biomedical Hypothesis Generation"**. JT-RAG jointly trains an Adaptive Neighborhood Evidence Selector (ANES) and a LoRA-adapted Qwen reasoner on temporal biomedical graphs.

## ⚙️ Installation

Run the following commands to create the environment and install the required dependencies:

```bash
conda create --name jtrag python=3.10
conda activate jtrag
pip install -r requirements.txt
```

`requirements-server.txt` is the original, unmodified server environment export. `requirements.txt` installs the packages used by this project and constrains their dependencies to the versions in that export, including PyTorch 2.8.0+cu128, Transformers 4.44.2, and PEFT 0.11.0. Packages in the full snapshot that this code does not require, such as `flash_attn`, are not installed by default. The CUDA 12.8 wheel source is already included, following the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2-8-0).

The installation targets a Linux NVIDIA GPU environment compatible with the exported CUDA build. The export does not record the server's Python version or NVIDIA driver; Python 3.10 above is the project setup choice, not a verified server version. A full installation and training run on a clean machine has not yet been verified.

After installation and data extraction, the two commands below run with built-in Immunotherapy defaults. No source edits, configuration files, or command-line options are required. Paths are resolved relative to the repository location, even when a script is launched from a different working directory. Models download automatically from Hugging Face on first use (network access required). For offline use, the scripts also automatically detect `models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/` and `models/Qwen2.5-1.5B-Instruct/` when complete model directories are placed there.

## 📂 Data Preparation

JT-RAG uses the same datasets as Bi-TLLM. This repository provides Immunotherapy as the preprocessing and training example.

1. **Download the data:** You can access and download the Immunotherapy dataset from the Hugging Face repository [here](https://huggingface.co/datasets/luzerwiki/Bi-TLLM).
2. **Extract the data:** Once downloaded, unzip the contents and place them into the `data/` directory in the root of this project. The dataset should contain `entity/entitylist.json`, `graph/`, and `negative/`, as shown below. The script also automatically accepts `graph_new/` instead of `graph/`; no path edits are needed.
3. **Preprocess the data:** Run the preprocessing script to construct training samples and weak evidence, then initialize PubMedBERT semantic embeddings and GraphSAGE structural embeddings:

```bash
python data/DatasetPreprocessing.py
```

```text
data/
├── DatasetPreprocessing.py
├── preprocess.py
├── semantic_embeddings.py
├── structural_embeddings.py
└── Immunotherapy/
    ├── entity/entitylist.json
    ├── graph/KWgraph_all_keys_{year}.edgelist
    └── negative/KWgraph_all_keys_{year}.edgelist
```

The script runs all three preprocessing stages in order and saves the results under `data/processed/Immunotherapy/`. The displayed command is run from the repository root. You can also enter `data/` and run `python DatasetPreprocessing.py`; outputs go to the same location.

The raw entity file maps entity IDs to mention records. Positive graph rows contain two entity IDs, a frequency, and a PMID list; negative rows contain two entity IDs. Snapshots use the years listed in `PREPROCESS_PARAMS["TIME_POINTS"]`. Each sample at year `T` retrieves neighbors and structural embeddings from the previous snapshot.

Weak evidence is the union of document, cross-side bridge, and historical common-neighbor evidence. The union is deduplicated, ranked by the existing historical-graph heuristic, and capped at 32 nodes (`K_INIT`).

For each sample year, preprocessing also writes `random/{year}.valid.tsv` and `random/{year}.test.tsv`, together with their compressed `.zst` versions. By default, the validation split contains up to 5,000 positive and 5,000 negative samples (seed 2026); all remaining samples form the test split. The two splits are disjoint and preserve the original sample IDs. Their sizes and seed can be changed at the top of `data/DatasetPreprocessing.py`.

## 🚀 Training

After preprocessing the data, start training directly with the built-in Immunotherapy parameters:

```bash
python JT-RAG.py
```

Training first initializes ANES with weak supervision at 2000, 2005, 2010, 2014, and 2017. For each year from 2019 through 2023, it refreshes ANES with weak supervision and alternates between Qwen LoRA tuning and ANES optimization with frozen-LLM feedback. The 2024 samples are not used for training.

After each joint training year, the model is evaluated on the **entire saved validation split of the next time point**: 2019→2020, 2020→2021, 2021→2022, 2022→2023, and 2023→2024. Predictions use the normalized Yes/No probability argmax. Per-sample predictions, Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC are saved under `outputs/Immunotherapy/validation/`, with combined JSON/CSV summaries. Evaluation does not sample from the full next-year dataset or read the test split. Missing or empty validation files raise an error. There is no early stopping or best-model selection.

Training continues to use `sample/{year}.tsv`. Validation is a rolling next-time evaluation: a year's data can subsequently enter training when the chronology advances to that year. It is not a permanently held-out split across all stages.

Checkpoints, LoRA adapters, training logs, and resume state are saved under `outputs/Immunotherapy/`. Rerunning the same command resumes completed stages by default. Use a new `OUTPUT_DIR` for a new experiment. Batch sizes, sampling sizes, epochs, and alternating rounds can be adjusted at the top of `JT-RAG.py`.

The example uses the original Immunotherapy V4 settings with three auxiliary loss coefficients disabled: `LAMBDA_LEN=0`, `LAMBDA_PSEUDO=0`, and `FEEDBACK_LAMBDA_ANCHOR=0`. The source reward penalties remain unchanged at `FEEDBACK_BETA_LENGTH=0` and `FEEDBACK_GAMMA_REDUNDANCY=0`, differing from the manuscript (`beta=0.05`, `gamma=0.10`). Semantic embeddings retain the source implementation's global entity descriptions.

## Repository Structure

```text
JT-RAG/
├── README.md
├── requirements.txt
├── requirements-server.txt      # Original full server environment snapshot
├── JT-RAG.py                    # Training entry point
├── data/
│   ├── DatasetPreprocessing.py  # Preprocessing entry point and parameters
│   ├── preprocess.py           # Historical graphs, samples and weak evidence
│   ├── semantic_embeddings.py  # PubMedBERT semantic embeddings
│   └── structural_embeddings.py # GraphSAGE structural embeddings
└── jtrag/                      # ANES, LoRA, LLM rewards and training utilities
```

The repository includes preprocessing, training, and next-time validation (`jtrag/validation.py`, `jtrag/eval_metrics.py`). There is no standalone test-set evaluation entry point. Downloaded datasets, generated embeddings, and model checkpoints are excluded from version control by `.gitignore`.
