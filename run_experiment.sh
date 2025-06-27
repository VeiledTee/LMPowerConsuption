#!/usr/bin/env bash

# Activate venv
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
pip install -r requirements.txt
which python
echo "venv activated and requirements installed."

# 1) Run main experiment
python src/experiment.py || { echo "experiment.py failed"; exit 1; }

# 2) Run the analysis
python src/analysis.py || { echo "analysis.py failed"; exit 1; }

# 3) Auto-commit and push results
git add results/*.csv results/*.md
git commit -m "Auto-update summary results on $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
git push origin master

