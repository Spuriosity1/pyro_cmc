import argparse
import os
import re
import sys
import numpy as np
import h5py

def load_file(path):
    with h5py.File(path, "r") as f:
        recip      = f["/geometry/recip_vectors"][:]   # (3,3) rows are b0,b1,b2
        index_cell = f["/geometry/index_cell"][:]

        raw          = f["/ssf/static_corr"][:]   # [n_corr, n_T, n_k, 2]
        corr_lookup  = [s.decode() if isinstance(s, bytes) else s
                        for s in f["/ssf/corr_lookup"][:]]
        sl_positions = f["/ssf/sl_positions"][:]  # [n_sl, 3] int64 (metadata)
        k_dims       = f["/ssf"].attrs["k_dims"][:].astype(int)   # [3]
        n_spins      = int(f["/ssf"].attrs["n_spins"])
        ssf_T        = f["/ssf/T_list"][:]        # sorted ascending
        n_ssf        = f["/ssf/n_samples"][:]

        # anneal/fieldcool write a thermodynamic /energy group; minimise emits a
        # single T=0 gradient-descent snapshot with no such group. Fall back to
        # the SSF's own temperature list and leave E/E2/n_E as NaN so the SSF
        # plotting path (which never touches them) still works on that output.
        if "energy" in f:
            T_list = f["/energy/T_list"][:]
            E      = f["/energy/E"][:]
            E2     = f["/energy/E2"][:]
            n_E    = f["/energy/n_samples"][:]
        else:
            T_list = ssf_T
            E = E2 = n_E = np.full(ssf_T.shape, np.nan)

    # static_corr is already the sublattice-contracted scalar S(q) (the phase
    # fold Σ_{mu,nu} exp(+i q·(r_mu-r_nu)) conj(Ã_mu) Ã_nu is done in the C++
    # writer), summed over MC samples.
    corr = raw[..., 0] + 1j * raw[..., 1]  # [n_corr, n_T, n_k]
    return (recip, index_cell,
            T_list, E, E2, n_E,
            corr, corr_lookup, sl_positions, k_dims, n_spins, ssf_T, n_ssf)


def normalize_ssf(corr, corr_lookup, k_dims, n_ssf):
    """Sample-normalise the contracted correlator and reshape to a k-grid.

    The sublattice phase contraction is performed in the C++ writer, so
    ``corr`` is already the scalar S(q) summed over samples.  Here we take the
    real part (S(q) is real for diagonal correlators), divide by the per-T
    sample count, and reshape flat k into (k0, k1, k2).

    Returns dict label -> array of shape [n_T, k0, k1, k2], real-valued.
    """
    contracted = np.real(corr)                       # [n_corr, n_T, n_k]
    contracted = contracted / n_ssf[np.newaxis, :, np.newaxis]

    k0, k1, k2 = k_dims
    contracted = contracted.reshape(len(corr_lookup), -1, k0, k1, k2)
    return {label: contracted[i] for i, label in enumerate(corr_lookup)}

def parse_params(path):
    """Extract key=value tokens from a filename stem into an ordered dict.

    Strips all extensions (e.g. .avg.h5).  Bare tokens are split at the
    first digit boundary, keeping a leading sign with the number:
    'b512' -> {'b': '512'}, 'K-0.002' -> {'K': '-0.002'}.
    Pure-alpha bare tokens like 'ec' are stored as {'ec': ''}.
    """
    stem = re.sub(r'(\.[a-zA-Z][a-zA-Z0-9]*)+$', '', os.path.basename(path))

    params = {}
    for token in stem.split("_"):
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
            params[k] = v
        else:
            m = re.match(r'^([A-Za-z]+)(-?\d.*)$', token)
            if m:
                params[m.group(1)] = m.group(2)
            else:
                params[token] = ''
    return params


def split_fixed_varying(files):
    """Return (fixed, varying_keys) across all filenames.

    fixed: dict of params whose value is identical in every file.
    varying_keys: list of params that differ across files (in order of first appearance).
    """
    all_params = [parse_params(f) for f in files]
    all_keys = dict.fromkeys(k for p in all_params for k in p)
    fixed = {
        k: all_params[0][k]
        for k in all_keys
        if all(p.get(k) == all_params[0].get(k) for p in all_params)
    }
    varying_keys = [k for k in all_keys if k not in fixed]
    return fixed, varying_keys, all_params


