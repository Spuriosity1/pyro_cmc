#!/usr/bin/env python3
"""Preprocessor: scan raw MC SSF outputs and write a compact Bragg summary CSV.

Meant to run on the cluster next to the raw ``*.out.h5`` files.  For every
sampled temperature of every input it records the three spiral star-arm
intensities (sorted Q1 >= Q2 >= Q3), the Γ-point intensity, n_spins, and every
``key=value`` parameter parsed from the filename (plus T).  The result is a
single ``<basename>_<timestamp>.bragg.csv`` that plot_QaB / plot_ratio /
plot_KT_phasedia can consume in place of the (much larger) raw files.

Usage:
    python3 scripts/extract_bragg.py RUN_DIR_OR_FILES... [-o out.bragg.csv]
"""

import argparse
import csv
import glob
import os
import sys
from datetime import datetime

from bragg import extract_raw_records
from plot_ssf import parse_params


def gather(paths):
    """Expand directory arguments into the ``*.out.h5`` files they contain."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.out.h5"))))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="*.out.h5 files or directories containing them")
    ap.add_argument("-o", "--output", default=None,
                    help="output CSV path (default: "
                         "<common-basename>_<timestamp>.bragg.csv)")
    args = ap.parse_args()

    files = gather(args.inputs)
    if not files:
        sys.exit("No input files found.")

    records = []
    for f in files:
        try:
            recs, _ = extract_raw_records(f, parse_params(f), uses_T=True)
        except Exception as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)
            continue
        records.extend(recs)

    if not records:
        sys.exit("No Bragg records extracted.")

    # Column order: filename parameters (first-seen), then the value columns.
    param_keys = []
    for r in records:
        for k in r['pm']:
            if k != 'T' and k not in param_keys:
                param_keys.append(k)
    cols = param_keys + ["T", "n_spins", "gamma", "Q1", "Q2", "Q3"]

    if args.output:
        out = args.output
    else:
        base = os.path.commonprefix([os.path.basename(f) for f in files])
        base = base.rstrip("_") or "run"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = f"{base}_{ts}.bragg.csv"

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = dict(r['pm'])
            row['n_spins'] = r['n_spins']
            row['gamma'] = r['I'][3]
            row['Q1'], row['Q2'], row['Q3'] = r['I'][0], r['I'][1], r['I'][2]
            w.writerow(row)

    print(f"Wrote {len(records)} rows to {out}")


if __name__ == "__main__":
    main()
