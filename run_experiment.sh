#!/usr/bin/env bash
cd /path/to/your/project
# 1) run main script
python experiment.py || { echo "main.py failed"; exit 1; }
# 2) then run the analysis
python analysis.py
# Auto-commit and push any changes (e.g. updated CSV/MD files)
git add results/*.csv results/*.md
git commit -m "Auto-update summary results on $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
git push origin main
