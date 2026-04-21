# LM‑PowerConsumption ⚡

Lightweight framework to evaluate the **accuracy ↔️ energy trade-off** of small and large language models (SLMs/LLMs) on QA benchmarks, with focus on how environmental impact and performance are intertwined.

Accepted to Canadian AI 2026 as a long paper!

## 🔍 Features

* Tracks energy usage and CO₂ emissions of LLM inference and document retrieval using CodeCarbon
* Supports Hugging Face and Ollama LLMs
* Two evaluation modes: **direct generation** (without retrieval) and **with retrieval** (Wikipedia-based)
* Computes common QA metrics: **Exact Match (EM)** and **F1**, alongside energy (kWh), emissions (kg CO₂), and runtime
* Resume support for long-running experiments

## 📁 Project Structure

```
.
├── src/
│   ├── cache/            # .pkl files representing the Wikipedia text used for retrieval
│   ├── config.py         # Experiment configuration (models, batch_size, modes, file paths)
│   ├── experiment.py     # Orchestrates experiment workflow
│   ├── inference.py      # Loads models & generates answers
│   ├── retrieval.py      # Implements Wikipedia document retrieval
│   ├── scorers.py        # Calculates EM & F1 metrics
│   ├── prompts.py        # Prompt templates for QA generation
│   └── utils.py          # Helper functions
├── requirements.txt      # Python dependencies
├── results/              # CSV outputs saved here
├── data/                 # Subset benchmarks
└── run_experiment.sh     # Full workflow: setup venv, experiment execution, analysis, and results versioning
```

## 🚀 Quick Start

1. **Clone & install**:

   ```bash
   git clone https://github.com/VeiledTee/LMPowerConsuption.git
   cd LMPowerConsuption
   pip install -r requirements.txt
   ```
2. **Configure** experiments in `config.py` (e.g., choose `model_candidates`, `modes`, paths, etc.)
3. **Run** the experiment with configuration set in step 2:

   ```bash
   python src/experiment.py
   ```
4. **Analyze results**:

   ```bash
   python src/analysis.py
   ```
5. **Run and analyze**
    ```bash
    chmod +x run_experiment.sh   # (one-time) make the script executable
    ./run_experiment.sh          # run the main pipeline and summary analysis
    ```

   Review EM, F1, total energy, emissions, and runtime

## 🎯 Use Cases

* Compare **model accuracy vs. energy footprint**
* Study **environmental cost** of retrieval‑augmented generation
* Facilitate reproducible energy‑aware NLP research
