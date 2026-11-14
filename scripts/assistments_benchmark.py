"""ASSISTments 2009-2010 benchmark for the standard IRT and BKT models.

Download the skill-builder file first (about 80 MB, not stored in this repository), e.g.:

    curl -L -o data/assistments_2009.csv \
        https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv

then run:

    python scripts/assistments_benchmark.py --data data/assistments_2009.csv

Writes results/assistments_benchmark.md and results/assistments_benchmark.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import assistments  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the ASSISTments skill-builder CSV")
    ap.add_argument("--items", type=int, default=50, help="most-answered problems used for IRT")
    ap.add_argument("--skills", type=int, default=10, help="most-practised skills used for BKT")
    ap.add_argument("--repeats", type=int, default=2, help="cross-validation repeats for BKT")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    df = assistments.load(a.data)
    r = assistments.benchmark(df, n_items=a.items, n_skills=a.skills, n_repeats=a.repeats)
    os.makedirs(a.outdir, exist_ok=True)
    md = assistments.report_markdown(r)
    with open(os.path.join(a.outdir, "assistments_benchmark.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(a.outdir, "assistments_benchmark.json"), "w") as fh:
        json.dump(r, fh, indent=2, default=float)
    print(md)
