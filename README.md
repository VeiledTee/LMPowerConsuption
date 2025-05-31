
# LMPowerConsuption

**LMPowerConsuption** is a lightweight framework for **measuring both the
accuracy *and* energy‑usage of small language models (SLMs) on popular QA
benchmarks**.  
It grew out of my PhD work on environmentally aware evaluation of retrieval‑
augmented generation systems.

---

## ✨ Key ideas
* **One‑liner experiments** – run any HF model on any HF dataset with a single
  config flag.  
* **Energy first‑class** – every prompt is wrapped in a
  [`codecarbon`](https://github.com/mlco2/codecarbon) tracker; per‑question
  Joules and aggregate kWh are logged automatically.  
* **Reproducible CSV outputs** – predictions, gold answers, EM / F1 and kWh are
  saved in tidy files ready for pandas/R analysis.

---

## 🔖 Repository layout
LMPowerConsuption/
├─ scripts/
│  ├─ hotpot\_smol\_eval\_scored.py   # HotpotQA evaluation & energy
│  └─ boolq\_smol\_eval\_scored.py    # BoolQ evaluation & energy
├─ Energy/                         # per‑question CodeCarbon logs
├─ avg\_results.txt                 # running table of overall scores
└─ requirements.txt                # pinned deps

---

## ⚡️ Quick start

```bash
# 1 — clone & create minimal env
git clone https://github.com/VeiledTee/LMPowerConsuption.git
cd LMPowerConsuption
python -m venv .venv && .venv/Scripts/activate   # Windows
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 2 — run a demo (BoolQ, question‑only prompt)
python scripts/boolq_smol_eval_scored.py

# 3 — check results
cat boolq_smol_q_only_results.csv      # per‑question data
cat avg_results.txt                    # dataset‑level summary
````

### Command‑line options

Each script can be tweaked via the constants at the top:

| Flag              | Meaning                                                      |
| ----------------- | ------------------------------------------------------------ |
| `INCLUDE_PASSAGE` |  Include dataset context/passage in the prompt               |
| `N_SAMPLES`       | Evaluate only the first *n* rows (speedy smoke‑run)          |
| `MODEL_NAME`      | Any HF identifier (quantised GGUF works via `ctransformers`) |
| `MAX_NEW_TOK`     | Decoding budget per question                                 |

---

## 📊 Output files

| File                            | What it contains                                |       |         |                            |
| ------------------------------- | ----------------------------------------------- | ----- | ------- | -------------------------- |
| `boolq_smol_q_only_results.csv` | `qid, predicted, gold, em, energy_kWh` per item |       |         |                            |
| `avg_results.txt`               | DATASET                                       | VERSION |MODEL | avg\_EM | avg\_energy\_kWh\` per run |
| `Energy/energy_<id>.csv`        | Raw `codecarbon` trace for the *id‑th* prompt   |       |         |                            |

Merge multiple runs with pandas or Excel to rank models by **energy per correct
answer**.

---

## 🔌 Adding a new benchmark

1. Copy one of the scripts in `scripts/`.
2. Change `DATASET_NAME`, `build_prompt()` and scoring function.
3. Log EM/F1 exactly as in Hotpot or BoolQ; `energy_sum` & `avg_results.txt`
   require no changes.

---

## 🛠 Requirements

\* Python ≥ 3.9
\* `transformers`, `datasets`, `codecarbon`, `accelerate`
\* NVIDIA GPU (optional) – int‑4 models fit in 6‑8 GB; CPU also works (slower).

Exact versions are pinned in `requirements.txt`.

---

## 🤝 Contributing

Pull requests that add new datasets, models, or alternative energy loggers
(`zeus`, `pyRAPL`, IPMI) are very welcome!  Please open an issue first to
discuss the scope.
