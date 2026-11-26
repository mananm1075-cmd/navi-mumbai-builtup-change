"""Reproduce everything:  python run_all.py
Needs the 10 yearly .s2clip.gz files in ./data (use unbundle.py for an .s2bundle)."""
import os
os.makedirs("results", exist_ok=True)
import classify_pooled
classify_pooled.main()          # normalisation + Random Forest for all years  (~4 min)
import make_outputs             # temporal consistency, area table, figures, validation points
