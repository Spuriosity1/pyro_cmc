#!/usr/bin/env python3
"""Shared Bragg/Γ-point extraction for the SSF post-processing scripts.

A *record* is the per-(file, temperature) summary that the peak-ratio and
peak-intensity plots consume::

    {
      'pm'        : {parsed filename params..., 'T': float},
      'n_spins'   : int,
      'I'         : [Q1, Q2, Q3, gamma],   # star arms sorted desc, then Γ
      'var_inter' : [.,.,.,.] or None,     # diag-summed, same order as I
      'var_intra' : [.,.,.,.] or None,
      'n_seeds'   : int or None,
      'n_per_seed': float,                 # NaN if unknown
    }

Records come from one of two sources, dispatched on the filename:

* ``<...>.out.h5`` — a raw MC output with the full SSF.  The three spiral
  star-arm intensities ``{(0,0,Q),(Q,0,0),(0,Q,0)}`` and Γ=(0,0,0) are read
  out, the arms sorted by intensity, and the per-seed variance arrays carried
  along for error bars.

* ``<...>.bragg.csv`` — a compact summary written by ``extract_bragg.py``.
  It is intentionally lossy: it keeps only the four intensities (plus the
  filename parameters, T and n_spins) and drops the variance arrays.  Plots
  fed from a .bragg.csv therefore fall back to seed-to-seed error bars (the
  ratio plots) or plain markers (the intensity plot).
"""

import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import h5py

from plot_ssf import load_file, normalize_ssf, parse_params


# ---------------------------------------------------------------------------
# q-index + error-bar helpers (shared by plot_QaB / plot_ratio)
# ---------------------------------------------------------------------------

def qz_to_idx(qz_frac, k_dim):
    return int(round(qz_frac * k_dim)) % k_dim


def se_from_inter(W_inter, n_seeds, n_spins):
    """Standard error of the multi-seed mean SSF from var_inter.

    W_inter : var_inter[k_q], summed over diagonal components — the biased
              (/K) population variance of the per-seed mean S(q) across seeds.
    Bessel-corrected SE of the K-seed mean: sqrt(W_inter / (K-1)) / n_spins.
    """
    if n_seeds is None or n_seeds < 2:
        return np.nan
    return np.sqrt(max(float(W_inter), 0.0) / (n_seeds - 1)) / n_spins


def se_from_intra(W_intra, n_seeds, n_per_seed, n_spins):
    """Standard error contribution to the multi-seed mean SSF from
    within-run (intra-seed) sampling noise.

    W_intra    : var_intra[k_q], summed over diagonal components -
                 Σ_seeds Var[single MC sample of S(q) | seed].
    n_per_seed : MC samples per seed at this temperature (= n_ssf_t / K).

    Treats samples within a seed as independent (no autocorrelation
    correction) and seeds as independent, consistent with se_from_inter().
    """
    if n_seeds is None or n_seeds < 1 or not n_per_seed or n_per_seed <= 0:
        return np.nan
    return np.sqrt(max(float(W_intra), 0.0) / (n_seeds**2 * n_per_seed)) / n_spins


def load_ssf_variance(path):
    """Load var_inter / var_intra / n_seeds from a merged HDF5 file (acc_runs.py).

    var_inter : (biased, /K) variance of the per-seed mean S(q) across seeds.
    var_intra : sum over seeds of the per-seed (within-run) single-sample
                variance of S(q).
    Both are already sublattice-contracted scalars, stored as
    [n_corr, n_T, n_k, 2] with the last axis carrying the Re/Im-part variances.
    Returns (var_inter, var_intra, n_seeds) as dicts label -> [n_T, k0,k1,k2]
    (real, = Re-part variance), or None for a variance whose dataset is absent.
    """
    def reshape_var(raw, corr_lookup, k_dims):
        real = raw[..., 0]                            # [n_corr, n_T, n_k]
        k0, k1, k2 = k_dims
        real = real.reshape(len(corr_lookup), -1, k0, k1, k2)
        return {label: real[i] for i, label in enumerate(corr_lookup)}

    var_inter = var_intra = None
    with h5py.File(path, "r") as f:
        ssf = f["/ssf"]
        k_dims = ssf.attrs["k_dims"][:].astype(int)
        corr_lookup = [s.decode() if isinstance(s, bytes) else s
                       for s in ssf["corr_lookup"][:]]
        if "var_inter" in ssf:
            var_inter = reshape_var(ssf["var_inter"][:], corr_lookup, k_dims)
        if "var_intra" in ssf:
            var_intra = reshape_var(ssf["var_intra"][:], corr_lookup, k_dims)
        energy = f["/energy"] if "energy" in f else None
        n_seeds = (int(energy["n_seeds"][()])
                   if energy is not None and "n_seeds" in energy else None)
    return var_inter, var_intra, n_seeds


# ---------------------------------------------------------------------------
# raw extraction
# ---------------------------------------------------------------------------

# CSV columns that are *not* filename parameters.
_VALUE_COLS = ("n_spins", "gamma", "Q1", "Q2", "Q3")


def extract_raw_records(fpath, file_params, uses_T, t_index=None):
    """Pull the Bragg/Γ records out of one raw ``*.out.h5`` file.

    Returns (records, title_T).  ``title_T`` is the single sampled temperature
    used when neither axis is T (else None).  An empty list is returned (with a
    warning) if the file lacks a Q tag or diagonal correlators.
    """
    qz_str = file_params.get('Q') or file_params.get('Qz')
    if qz_str is None:
        print(f"Warning: 'Q'/'Qz' not found in {os.path.basename(fpath)}, skipping",
              file=sys.stderr)
        return [], None
    qz = float(qz_str)

    (_, _, _, _, _, _,
     corr, corr_lookup, sl_positions, k_dims, n_spins, ssf_T, n_ssf) = load_file(fpath)

    S = normalize_ssf(corr, corr_lookup, k_dims, n_ssf)
    diag = [c for c in ("xx", "yy", "zz") if c in S]
    if not diag:
        print(f"Warning: no diagonal correlators in {os.path.basename(fpath)}, skipping",
              file=sys.stderr)
        return [], None

    var_inter, var_intra, n_seeds = load_ssf_variance(fpath)

    # Qz is in units of 2π/a_cubic; k_dims[i] = L for cubic supercell.
    qi = qz_to_idx(qz, k_dims[0])
    arms = [(0, 0, qi), (qi, 0, 0), (0, qi, 0)]   # the three spiral star arms
    gamma = (0, 0, 0)

    n_T = corr.shape[1]
    title_T = None
    if uses_T:
        t_indices = range(n_T)
    else:
        t_idx0 = t_index if t_index is not None else n_T - 1
        if not (0 <= t_idx0 < n_T):
            sys.exit(f"--t-index {t_idx0} out of range [0, {n_T - 1}]")
        t_indices = [t_idx0]
        title_T = float(ssf_T[t_idx0])

    records = []
    for t_idx in t_indices:
        pm = dict(file_params)
        pm['T'] = float(ssf_T[t_idx])

        def I(q):
            return sum(S[c][t_idx, q[0], q[1], q[2]] for c in diag)

        # Rank the three arms by intensity at THIS temperature (see plot_ratio
        # for why per-T rather than once at a reference T), then pin Γ last.
        order_q = sorted(arms, key=I, reverse=True) + [gamma]

        def vsum(d, q):
            return sum(d[c][t_idx, q[0], q[1], q[2]] for c in diag)

        vi = ([vsum(var_inter, q) for q in order_q]
              if var_inter is not None and n_seeds is not None else None)
        va = ([vsum(var_intra, q) for q in order_q]
              if var_intra is not None and n_seeds is not None else None)

        records.append({
            'pm': pm,
            'n_spins': n_spins,
            'I': [I(q) for q in order_q],
            'var_inter': vi,
            'var_intra': va,
            'n_seeds': n_seeds,
            'n_per_seed': n_ssf[t_idx] / n_seeds if n_seeds else np.nan,
        })

    return records, title_T


# ---------------------------------------------------------------------------
# .bragg.csv extraction
# ---------------------------------------------------------------------------

def read_bragg_records(fpath, uses_T, t_index=None):
    """Read records from a pre-extracted ``*.bragg.csv`` summary.

    Returns (records, title_T).  The CSV has one row per (original file,
    temperature), written in the raw temperature-index order; when neither
    axis is T we regroup the rows by their non-T parameters and keep a single
    temperature per run (the ``t_index``-th, default last), mirroring the raw
    path.
    """
    rows = []
    with open(fpath, newline="") as fh:
        for row in csv.DictReader(fh):
            pm = {}
            for k, v in row.items():
                if k in _VALUE_COLS or v is None or v == "":
                    continue
                pm[k] = float(v) if k == 'T' else v
            rows.append({
                'pm': pm,
                'n_spins': int(float(row['n_spins'])) if row.get('n_spins') else None,
                'I': [float(row['Q1']), float(row['Q2']),
                      float(row['Q3']), float(row['gamma'])],
                'var_inter': None,
                'var_intra': None,
                'n_seeds': None,
                'n_per_seed': np.nan,
            })

    if uses_T:
        return rows, None

    # Collapse each run (same params bar T) to a single temperature.
    groups = OrderedDict()
    for r in rows:
        key = tuple(sorted((k, v) for k, v in r['pm'].items() if k != 'T'))
        groups.setdefault(key, []).append(r)

    selected, title_T = [], None
    for grp in groups.values():
        idx = t_index if t_index is not None else len(grp) - 1
        if not (0 <= idx < len(grp)):
            sys.exit(f"--t-index {idx} out of range [0, {len(grp) - 1}]")
        selected.append(grp[idx])
        title_T = grp[idx]['pm'].get('T')
    return selected, title_T


# ---------------------------------------------------------------------------
# unified loader
# ---------------------------------------------------------------------------

def _filter_inputs(files):
    """Drop unreadable HDF5 inputs; CSV summaries are passed straight through."""
    clean = []
    for f in files:
        if f.endswith(".csv"):
            clean.append(f)
            continue
        try:
            with h5py.File(f):
                clean.append(f)
        except Exception as e:
            print(f"File {f} corrupt: {e}", file=sys.stderr)
    return clean


def fixed_from_records(records):
    """Parameters (excluding T) whose value is identical across every record."""
    keys = []
    for r in records:
        for k in r['pm']:
            if k != 'T' and k not in keys:
                keys.append(k)
    fixed = {}
    for k in keys:
        vals = set(r['pm'].get(k) for r in records)
        if len(vals) == 1 and None not in vals:
            fixed[k] = next(iter(vals))
    return fixed


def load_records(files, uses_T, t_index=None):
    """Load Bragg records from a mix of ``*.out.h5`` and ``*.bragg.csv`` inputs.

    Returns (records, title_T, fixed), where ``fixed`` is the dict of
    parameters constant across all records (for plot titles).
    """
    records, title_T = [], None
    for f in _filter_inputs(files):
        if f.endswith(".csv"):
            recs, tT = read_bragg_records(f, uses_T, t_index)
        else:
            recs, tT = extract_raw_records(f, parse_params(f), uses_T, t_index)
        records.extend(recs)
        if tT is not None:
            title_T = tT
    return records, title_T, fixed_from_records(records)
