#!/usr/bin/env bash

# Activate venv (relative to current dir)
source .venv/bin/activate

# 1) Run main experiment
python experiment.py || { echo "experiment.py failed"; exit 1; }

# 2) Run the analysis
python analysis.py || { echo "analysis.py failed"; exit 1; }

# 3) Auto-commit and push results
git add results/*.csv results/*.md
git commit -m "Auto-update summary results on $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
git push origin master
