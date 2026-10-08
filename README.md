# JT-RAG: A Jointly Trained Temporal Retrieval-Augmented Generation Framework for Biomedical Hypothesis Generation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains the implementation of **JT-RAG: A Jointly Trained Temporal Retrieval-Augmented Generation Framework for Biomedical Hypothesis Generation**.

## ⚙️ Installation

Run the following commands to create the environment and install the required dependencies:

```bash
conda create --name jtrag python=3.10
conda activate jtrag
pip install -r requirements-server.txt
```

## 📂 Data Preparation

The datasets required for training and evaluation are hosted on Hugging Face.

1. **Download the data:** You can access and download all datasets from our Hugging Face repository [here](https://huggingface.co/datasets/luzerwiki/Bi-TLLM).
2. **Extract the data:** Once downloaded, unzip the contents and place them into the `data/` directory in the root of this project.
3. **Preprocess the data:** Run the preprocessing script to format the datasets for the model:

```bash
python data/DatasetPreprocessing.py
```

## 🚀 Training

After preprocessing, run the following command to start training:

```bash
python JT-RAG.py
```
