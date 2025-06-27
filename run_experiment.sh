#!/usr/bin/env bash
cd /path/to/your/project
# 1) run main script
python experiment.py || { echo "main.py failed"; exit 1; }
# 2) then run the analysis
python analysis.py
