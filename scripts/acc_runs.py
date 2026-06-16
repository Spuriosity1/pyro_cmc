#!/usr/bin/env python3

import re
import sys
import os.path
import numpy as np
import h5py

# Usage: acc_heat_capacity L=8_J1=..._seed=<hex>_Tc=<val>_.out.h5 ...
#
# All input files must share the same L, J1, J2, J3, Tc (differ only in seed).
# Merges across seeds and computes the cross-seed variance of the energy
# per primitive cell, e = E_total / N_primitive_cells.
#
# Assumes all seeds use identical --n_sample (uniform n_per_seed per temperature).
#
# Writes:
#   <params>_mergeK_.avg.h5  — merged over K seeds
#
# Datasets in /energy:
#   T_list    — temperature grid
#   E         — sum of e over seeds (e = energy per primitive cell)
#   E2        — sum of e² over seeds
#   n_samples — total sample count (= K, since each seed contributes 1 per T)
#   e_mean    — cross-seed mean energy per primitive cell
#   e2_mean   — cross-seed mean of e²
#   var       — cross-seed variance: e2_mean - e_mean²
#
# /geometry:
#   index_cell
#   recip_vectors
#
# Datasets in /ssf:
#   static_corr      — Σ_seeds raw sum of per-sample correlators [n_corr,n_T,n_k,n_sl,n_sl,2]
#   static_corr_2    — Σ_seeds (per-seed raw sum)²  [same shape]  ← inter-seed 2nd moment
#   n_samples        — total sample count across seeds [n_T]
#   var_inter        — inter-seed variance of C_{μν}(k): Var_seeds[mean_C_seed] [same shape]
#   var_intra        — sum of per-seed intrinsic variances of C_{μν}(k) [same shape]
#                      (absent if no seed file contained static_corr_2)
#   T_list, corr_lookup, sl_positions, attrs — copied from first file
#
# Last dim of all corr arrays is [Re, Im], storing Re and Im variances separately.
#
# Heat capacity per spin: C/N = var * N_prim / (T² * N_spins)
#                              = var / (T² * 16)
# is left to post-processing.


def load_energy_raw(fname):
    """Return T_list, E_sum, E2_sum, n_samples (raw sums from energy_manager)."""
    with h5py.File(fname, "r") as f:
        g = f["/energy"]
        assert isinstance(g, h5py.Group)
        T  = np.array(g["T_list"])
        E  = np.array(g["E"])
        E2 = np.array(g["E2"])
        n  = np.array(g["n_samples"])
    return T, E, E2, n

def load_geometry(fname):
    with h5py.File(fname, "r") as f:
        g= f["/geometry"]
        assert isinstance(g, h5py.Group)
        idx_cell = np.array(g["index_cell"])
        recip_vectors = np.array(g["recip_vectors"])

    return idx_cell, recip_vectors


"""
expected layout:
/ssf/T_list              Dataset {1}
/ssf/corr_lookup         Dataset {3}
/ssf/n_samples           Dataset {1}
/ssf/sl_positions        Dataset {16, 3}
/ssf/static_corr         Dataset {3, 1, 512, 16, 16, 2}
"""
def load_ssf_raw(fname):
    with h5py.File(fname, "r") as f:
        g= f["/ssf"]
        assert isinstance(g, h5py.Group)
        T  = np.array(g["T_list"])
        corr_lookup = np.array(g["corr_lookup"])
        n_samples = np.array(g["n_samples"])
        sl_pos = np.array(g["sl_positions"])
        static_corr  = np.array(g["static_corr"])
        static_corr_2 = np.array(g["static_corr_2"]) if "static_corr_2" in g else None
        attrs = {k: v for k, v in g.attrs.items()}

    return T, corr_lookup, n_samples, sl_pos, static_corr, static_corr_2, attrs

_COMPATIBLE_TAGS = {
    "L":   (r"L=(\d+)_",              int),
    "J1":  (r"J1=([-\d.e+]+)_",       float),
    "J2":  (r"J2=([-\d.e+]+)_",       float),
    "J3":  (r"J3=([-\d.e+]+)_",       float),
    "Tc": (r"Tc=([-\d.e+]+)_",      float),
}


def parse_tags(fname):
    base = os.path.basename(fname)
    tags = {}
    for name, (pattern, cast) in _COMPATIBLE_TAGS.items():
        m = re.search(pattern, base)
        if m is None:
            raise ValueError(f"No {name}=<val> tag in: {base}")
        tags[name] = cast(m.group(1))
    return tags

def check_compatibility(fnames):
    ref = parse_tags(fnames[0])
    for fname in fnames[1:]:
        tags = parse_tags(fname)
        mismatches = [
            f"{k}: {ref[k]} vs {tags[k]}"
            for k in ref if ref[k] != tags[k]
        ]
        if mismatches:
            raise ValueError(
                f"Incompatible tags in {os.path.basename(fname)}: "
                + ", ".join(mismatches)
            )


def merged_name(representative_file, n_seeds):
    """Replace seed=<hex>_ with mergeN_ and change extension to .avg.h5."""
    base = os.path.basename(representative_file)
    out = re.sub(r"seed=[0-9a-fA-F]+_", f"merge{n_seeds}_", base)
    out = re.sub(r"\.out\.h5$", ".avg.h5", out)
    if out == base:
        raise ValueError(f"Could not generate output name from: {base}")
    return os.path.join(os.path.dirname(representative_file), out)


def main(fnames):
    if not fnames:
        print("Usage: acc_heat_capacity L=8_J1=..._seed=<hex>_T_c=<val>_.out.h5 ...")
        sys.exit(1)

    check_compatibility(fnames)
    tags = parse_tags(fnames[0])

    T_ref    = None
    E_total  = None
    E2_total = None
    n_total  = None

    # accumulate the energies
    for fname in fnames:
        T, E, E2, n = load_energy_raw(fname)
        if T_ref is None:
            T_ref    = T
            E_total  = E.copy()
            E2_total = E2.copy()
            n_total  = n.copy()
        else:
            if not np.allclose(T, T_ref):
                raise ValueError(f"T_list mismatch in {fname}")
            E_total  += E
            E2_total += E2
            n_total  += n

    # accumulate the ssf's
    T_ref_ssf = None
    corr_lookup_ref=None
    sl_positions_ref=None
    ssf_attrs_ref = None

    total_static_corr = None
    total_static_corr_2 = None   # Σ_seeds C_s²  (inter-seed 2nd moment)
    total_intrinsic_corr_2 = None  # Σ_seeds static_corr_2[seed]  (intra-seed 2nd moment)
    total_static_samp = None

    for fname in fnames:
        T, corr_lookup, n_samp, sl_pos, static_corr, static_corr_2, attrs = load_ssf_raw(fname)
        if T_ref_ssf is None:
            T_ref_ssf = T
            corr_lookup_ref = corr_lookup
            sl_positions_ref = sl_pos
            ssf_attrs_ref = attrs

            total_static_corr = static_corr.copy()
            total_static_corr_2 = static_corr**2
            total_static_samp = n_samp.copy()
            total_intrinsic_corr_2 = static_corr_2.copy() if static_corr_2 is not None else None
        else:
            if not np.allclose(T_ref_ssf, T):
                raise RuntimeError(f"Temperature range of file {fname} incompatible")
            if np.any(corr_lookup_ref != corr_lookup):
                raise RuntimeError(f"Stored correlators of file {fname} incompatible")
            if not np.allclose(sl_positions_ref, sl_pos):
                raise RuntimeError(f"Sublattice convention incompatible in file {fname}")
            attr_mismatches = [
                f"{k}: {ssf_attrs_ref[k]!r} vs {attrs[k]!r}"
                for k in ssf_attrs_ref
                if k not in attrs or not np.array_equal(ssf_attrs_ref[k], attrs[k])
            ] + [f"{k}: missing in reference" for k in attrs if k not in ssf_attrs_ref]
            if attr_mismatches:
                raise RuntimeError(
                    f"SSF group attribute mismatch in {fname}: "
                    + ", ".join(attr_mismatches)
                )

            total_static_corr += static_corr
            total_static_corr_2 += static_corr**2
            total_static_samp += n_samp
            if total_intrinsic_corr_2 is not None:
                if static_corr_2 is not None:
                    total_intrinsic_corr_2 += static_corr_2
                else:
                    print(f"Warning: {fname} has no static_corr_2; dropping var_intra",
                          file=sys.stderr)
                    total_intrinsic_corr_2 = None
            

    K = len(fnames)
    e_mean  = E_total  / n_total
    e2_mean = E2_total / n_total
    var     = e2_mean - e_mean**2

    # SSF variance terms.
    # Assumes uniform n_per_seed across seeds; n_per_seed[t] = total_samp[t] / K.
    # Both var arrays have shape [n_corr, n_T, n_k, n_sl, n_sl, 2] where the last
    # dimension stores [Re variance, Im variance] independently.
    n_per_seed = total_static_samp / K   # [n_T]
    # broadcast over [n_corr, n_T, n_k, n_sl, n_sl, 2]
    n_ = n_per_seed[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    # Inter-seed variance: Var_seeds[mean_C_seed]
    #   = (1/K) Σ_s (C_s/n)² − ((1/K) Σ_s C_s/n)²
    #   = total_corr2/(K·n²) − (total_corr/(K·n))²
    ssf_var_inter = (total_static_corr_2 / (K * n_**2)
                     - (total_static_corr / (K * n_))**2)

    # Sum of per-seed intrinsic variances: Σ_s Var_intra_s
    #   = Σ_s [C2_s/n − (C_s/n)²]
    #   = total_intr_C2/n − total_corr2/n²
    if total_intrinsic_corr_2 is not None:
        ssf_var_intra = total_intrinsic_corr_2 / n_ - total_static_corr_2 / n_**2
    else:
        ssf_var_intra = None

    idx, rcv = load_geometry(fnames[0])

    out = merged_name(fnames[0], K)
    with h5py.File(out, "w") as f:
        g = f.create_group("energy")
        g.create_dataset("T_list",    data=T_ref)
        g.create_dataset("E",         data=E_total)
        g.create_dataset("E2",        data=E2_total)
        g.create_dataset("n_samples", data=n_total)
        g.create_dataset("e_mean",    data=e_mean)
        g.create_dataset("e2_mean",   data=e2_mean)
        g.create_dataset("var",       data=var)
        g.create_dataset("n_seeds",   data=np.uint64(K))

        gg = f.create_group("geometry")
        gg.create_dataset("index_cell",  data=idx)
        gg.create_dataset("recip_vectors", data=rcv)

        sg = f.create_group("ssf")
        sg.create_dataset("T_list",        data=T_ref_ssf)
        sg.create_dataset("corr_lookup",   data=corr_lookup_ref)
        sg.create_dataset("n_samples",     data=total_static_samp)
        sg.create_dataset("static_corr",   data=total_static_corr)
        sg.create_dataset("static_corr_2", data=total_static_corr_2)
        sg.create_dataset("var_inter",     data=ssf_var_inter)
        if ssf_var_intra is not None:
            sg.create_dataset("var_intra", data=ssf_var_intra)
        sg.create_dataset("sl_positions",  data=sl_positions_ref)
        for k, v in ssf_attrs_ref.items():
            sg.attrs[k] = v

    print(f"{K} seed(s) merged → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
