#!/usr/bin/env bash

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv || { echo "Failed to create venv"; exit 1; }
    echo "Virtual environment created"
fi

# Activate venv
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
echo "Virtual environment activated"

# Upgrade pip first
pip install --upgrade pip || { echo "pip upgrade failed"; exit 1; }

# Install requirements
pip install -r requirements.txt || { echo "Requirements installation failed"; exit 1; }
echo "Dependencies installed"

# Verify Python path
which python
echo "Python path confirmed"

# 1) Run main experiment
python src/experiment.py || { echo "experiment.py failed"; exit 1; }

# 2) Run the analysis
python src/analysis.py || { echo "analysis.py failed"; exit 1; }

# 3) Auto-commit and push results
git add results/*.csv results/*.md
git commit -m "Auto-update summary results on $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
git push origin master