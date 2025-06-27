# LM‑PowerConsumption ⚡

Lightweight framework to evaluate the **accuracy ↔️ energy trade-off** of small and large language models (SLMs/LLMs) on QA benchmarks, with focus on environmental impact and performance.

## 🔍 Features

* Tracks energy usage and CO₂ emissions using CodeCarbon
* Supports multiple LLMs (e.g., Gemma, Llama‑2)
* Two evaluation modes: **with retrieval** (Wikipedia-based) and **direct generation**
* Computes common QA metrics: **Exact Match (EM)** and **F1**, alongside energy (kWh), emissions (kg CO₂), and runtime
* Resume support for long-running experiments ([github.com][1])

## 📁 Project Structure

```
.
├── config.py        # Experiment configuration (models, batch_size, modes, file paths)
├── experiment.py    # Orchestrates experiment workflow
├── inference.py     # Loads models & generates answers
├── retrieval.py     # Implements Wikipedia document retrieval
├── scorers.py       # Calculates EM & F1 metrics
├── prompts.py       # Prompt templates for QA generation
├── utils.py         # Helper functions
├── requirements.txt # Python dependencies
├── results/         # CSV outputs saved here
└── data/            # QA benchmark & Wikipedia resources
```

## 🚀 Quick Start

1. **Clone & install**:

   ```bash
   git clone https://github.com/VeiledTee/LMPowerConsuption.git
   cd LMPowerConsuption
   pip install -r requirements.txt
   ```
2. **Configure** experiments in `config.py` (e.g., choose `model_candidates`, `modes`, paths, etc.)
3. **Run** the experiment:

   ```bash
   python experiment.py
   ```
4. **Analyze results**:

   ```python
   import pandas as pd
   df = pd.read_csv("results/energy/*.csv")
   print(df[["f1", "inference_energy"]].mean())
   ```
5. **Run and analyze**
    ```bash
    chmod +x run_all.sh  # (one-time) make the script executable
    ./run_experiment.sh          # run the main pipeline and summary analysis
    ```

   Review EM, F1, total energy, emissions, and runtime ([github.com][1])

## 🎯 Use Cases

* Compare **model accuracy vs. energy footprint**
* Study **environmental cost** of retrieval‑augmented generation
* Facilitate reproducible energy‑aware NLP research

## 📌 Requirements

* Python 3.11+
* PyTorch (with GPU/CUDA support recommended)
* `CodeCarbon` for energy tracking
* Access to required QA dataset & local Wikipedia index