# Energy-Efficient QA System

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project evaluates the energy consumption of large language models on the HotpotQA benchmark, comparing different configurations and retrieval approaches.

## 📋 Features

- Energy consumption tracking with CodeCarbon
- Support for multiple LLMs (Gemma, Llama-2)
- Two evaluation modes: with and without retrieval
- Wikipedia-based document retrieval system
- Comprehensive metrics: EM, F1, energy, emissions
- Resume functionality for long-running experiments

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA if available)

### Installation
```bash
git clone https://github.com/VeiledTee/LMPowerConsuption.git
cd LMPowerConsuption
pip install -r requirements.txt
```

### Configuration
Edit `config.py` to:
- Select models (`model_candidates`)
- Adjust experiment parameters (`batch_size`, `max_new_tokens`)
- Set paths (`wiki_dir`, `energy_dir`)
- Choose modes (`modes`)

### Running the Experiment
```bash
python main.py
```

### Expected Output
```
2023-10-15 14:30:00 - energy_eval - INFO - Starting experiment with config:...
2023-10-15 14:30:01 - energy_eval - INFO - Loaded dataset with 1000 samples
2023-10-15 14:30:05 - energy_eval - INFO - Running model: google/gemma-2b-it
2023-10-15 14:30:10 - energy_eval - INFO - Starting q mode for google/gemma-2b-it
100%|████████████████████████| 1000/1000 [15:20<00:00, 1.08s/sample]
2023-10-15 14:45:30 - energy_eval - INFO - Completed q mode for google/gemma-2b-it
...
```

Results will be saved in CSV format to `results/energy/`.

## 🧩 Project Structure
```
energy-efficient-qa/
├── config.py             # Experiment configuration
├── main.py               # Main orchestration script
├── inference.py          # Model loading and generation
├── prompts.py            # Prompt engineering
├── retrieval.py          # Wikipedia retrieval system
├── scorers.py            # Evaluation metrics (EM, F1)
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
├── README.md             # This document
└── results/              # Output directory (auto-created)
    └── energy/           # Energy and performance results
```

## 📊 Results Analysis
Results include:
- Question ID and text
- Model predictions and gold answers
- Exact Match and F1 scores
- Energy consumption (kWh)
- CO2 emissions (kg)
- Processing duration (seconds)

Use Pandas for analysis:
```python
import pandas as pd

df = pd.read_csv("results/energy/hotpot_gemma-2b-it_q+r.csv")
mean_f1 = df["f1"].mean()
total_energy = df["inference_energy"].sum()
print(f"Average F1: {mean_f1:.2f}, Total Energy: {total_energy:.4f} kWh")
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- HotpotQA dataset providers
- Hugging Face Transformers library
- CodeCarbon for emissions tracking