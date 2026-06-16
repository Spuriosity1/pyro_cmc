#pragma once

#include "MC.hpp"
#include "abstract_manager.hpp"
#include <lattice_lib/fourier.hpp>
#include "H5Apublic.h"
#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Tpublic.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


// Scalar accessor for spin component at compile-time offset (0=x, 1=y, 2=z).
template<int Offset>
struct OffsetSpinAccessor {
    static_assert(Offset >= 0 && Offset < 3, "offset must be 0, 1 or 2");
    static double get(const CMC::HeisenbergSpin& s) { return s.S[Offset]; }
    static void   set(CMC::HeisenbergSpin& s, double v) { s.S[Offset] = v; }
};

using FT_Sx_t = FourierTransformC2C<CMC::HeisenbergSpin,
    OffsetSpinAccessor<0>, CMC::HeisenbergSpin>;
using FT_Sy_t = FourierTransformC2C<CMC::HeisenbergSpin,
    OffsetSpinAccessor<1>, CMC::HeisenbergSpin>;
using FT_Sz_t = FourierTransformC2C<CMC::HeisenbergSpin,
    OffsetSpinAccessor<2>, CMC::HeisenbergSpin>;


struct CorrSpec {
    int alpha, beta;   // 0=x, 1=y, 2=z
    std::string label; // e.g. "xy"
};


// Measures a user-specified set of sublattice-resolved spin component correlators.
//
// Accumulates, per temperature and per requested (α,β) pair:
//
//   C^{αβ}_{μν}(q) = Σ_samples  conj(Ã_μ^α(q)) · Ã_ν^β(q)
//   and the error term,
//   sq_C^{αβ}_{μν}(q) = Σ_samples  conj(Ã_μ^α(q)) · Ã_ν^β(q) ^2 entrywise as real and imaginary parts
//
//
// where Ã_μ^α is the raw per-cell DFT of spin component α on sublattice μ
// (sublattice phase exp(−i q·r_μ) NOT included; apply in post-processing via
// SublatWeightMatrix::phase_factors()).
//
// HDF5 output layout (write_group):
//   static_corr  [n_corr, n_T, n_k, n_sl, n_sl, 2]  — last dim = re/im
//   static_corr_2  [n_corr, n_T, n_k, n_sl, n_sl, 2]
//   corr_lookup  [n_corr]                             — VL strings, e.g. "xx", "xz"
//   T_list       [n_T]                                — temperatures, sorted ascending
//   n_samples    [n_T]
//   sl_positions [n_sl, 3]                            — int64 sublattice coords
//   attribute n_spins                                 — total spin count
//   attribute k_dims[3]                               — k-space grid dimensions
//
// Usage:
//   ssf_manager ssf(lat, {"xx", "yy", "zz"});
//   ssf.new_T(T);
//   for (...) { mc.sweep(...); ssf.sample(); }
//   ssf.write_group(file_id, "/ssf");
class ssf_manager : public abstract_manager {
    // Only the needed component FTs are constructed (avoids unnecessary FFTW plans).
    std::optional<FT_Sx_t> ft_x_;
    std::optional<FT_Sy_t> ft_y_;
    std::optional<FT_Sz_t> ft_z_;

    std::vector<CorrSpec> corr_specs_;
    // corr_[corr_idx][temp_idx] — grows via on_new_temp()
    std::vector<std::vector<FourierCorrelator<CMC::HeisenbergSpin>>> corr_;
    std::vector<std::vector<FourierCorrelator<CMC::HeisenbergSpin>>> corr_sq_;

    int n_sl_;
    int n_kpoints_;
    int n_spins_;
    ivec3_t k_dims_;
    std::vector<ipos_t> sl_positions_;

    const bool store_error_term_;

    void on_new_temp() override {
        for (auto& v : corr_)
            v.emplace_back(n_sl_, k_dims_);
        for (auto& v : corr_sq_)
            v.emplace_back(n_sl_, k_dims_);
    }

public:
    // sc must outlive this object.
    // correlator_names: 2-char component-pair strings, e.g. {"xx","yy","zz","xy"}.
    ssf_manager(CMC::Lattice& sc,
                const std::vector<std::string>& correlator_names,
                size_t n_temperatures_reserve = 0, bool store_error_term=true)
        : n_sl_(static_cast<int>(
              std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions).size())),
          n_kpoints_(sc.lattice.num_primitive_cells()),
          n_spins_(static_cast<int>(sc.get_objects<CMC::HeisenbergSpin>().size())),
          k_dims_(sc.lattice.size()),
          sl_positions_(
              std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions).begin(),
              std::get<SlPos<CMC::HeisenbergSpin>>(sc.sl_positions).end()),
          store_error_term_(store_error_term)
    {
        T_list.reserve(n_temperatures_reserve);
        n_samples.reserve(n_temperatures_reserve);

        auto axis_idx = [](char c) -> int {
            if (c == 'x') return 0;
            if (c == 'y') return 1;
            if (c == 'z') return 2;
            throw std::invalid_argument(
                std::string("ssf_manager: unknown spin component '") + c + "'");
        };

        bool need[3] = {false, false, false};
        for (const auto& name : correlator_names) {
            if (name.size() != 2)
                throw std::invalid_argument(
                    "ssf_manager: correlator name must be 2 chars: " + name);
            int a = axis_idx(name[0]);
            int b = axis_idx(name[1]);
            corr_specs_.push_back({a, b, name});
            need[a] = true;
            need[b] = true;
        }

        if (need[0]) ft_x_.emplace(sc);
        if (need[1]) ft_y_.emplace(sc);
        if (need[2]) ft_z_.emplace(sc);

        corr_.resize(corr_specs_.size());
        corr_sq_.resize(corr_specs_.size());
        for (auto& v : corr_) v.reserve(n_temperatures_reserve);
        for (auto& v : corr_sq_) v.reserve(n_temperatures_reserve);
    }

    // Fourier-transform the current spin configuration and accumulate C^{αβ}_{μν}(q).
    void sample() {
        assert(!T_list.empty());
        if (ft_x_) ft_x_->transform();
        if (ft_y_) ft_y_->transform();
        if (ft_z_) ft_z_->transform();

        auto buf = [&](int axis) -> const FourierBuffer<CMC::HeisenbergSpin>& {
            if (axis == 0) return ft_x_->get_buffer();
            if (axis == 1) return ft_y_->get_buffer();
            return ft_z_->get_buffer();
        };

        for (size_t c = 0; c < corr_specs_.size(); ++c) {
            auto this_corr = correlate(buf(corr_specs_[c].alpha),
                    buf(corr_specs_[c].beta));
            // calculate the dirrect correlator
            corr_[c][curr_idx] += this_corr;

            if (store_error_term_){
                // ---- error term: compute entrywise squares of all terms in corr_specs
                for (int i=0; i<n_sl_; i++)
                    for (int j=0; j<n_sl_; j++)
                        for (auto& C_k : this_corr(i,j))
                            C_k= {C_k.real()*C_k.real(), C_k.imag()*C_k.imag()};

                corr_sq_[c][curr_idx] += this_corr;
            }
        }

        n_samples[curr_idx]++;
    }

    void write_group(hid_t file_id, const char* group_name = "/ssf") override;
};


inline void ssf_manager::write_group(hid_t file_id, const char* group_name) {
    const size_t  n_T    = T_list.size();
    const size_t  n_corr = corr_specs_.size();
    const hsize_t ns     = static_cast<hsize_t>(n_sl_);
    const hsize_t nk     = static_cast<hsize_t>(n_kpoints_);

    // Sort temperatures ascending
    std::vector<size_t> ord(n_T);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](size_t a, size_t b) { return T_list[a] < T_list[b]; });

    std::vector<double> sT(n_T);
    std::vector<size_t> sN(n_T);
    for (size_t i = 0; i < n_T; ++i) {
        sT[i] = T_list[ord[i]];
        sN[i] = n_samples[ord[i]];
    }

    // Open or create group
    hid_t grp;
    if (H5Lexists(file_id, group_name, H5P_DEFAULT) > 0)
        grp = H5Gopen2(file_id, group_name, H5P_DEFAULT);
    else
        grp = H5Gcreate2(file_id, group_name,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (grp < 0)
        throw std::runtime_error(
            std::string("ssf_manager: failed to open/create group ") + group_name);

    auto write_1d = [&](const char* name, hid_t type, hsize_t len, const void* data) {
        hid_t sp = H5Screate_simple(1, &len, nullptr);
        hid_t ds = H5Dcreate2(grp, name, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) {
            H5Sclose(sp);
            H5Gclose(grp);
            throw std::runtime_error(
                std::string("ssf_manager: failed to create ") + name);
        }
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(sp);
    };

    // --- static_corr [n_corr, n_T, n_k, n_sl, n_sl, 2] ---
    // --- static_corr_2 [n_corr, n_T, n_k, n_sl, n_sl, 2] ---
    {
        hsize_t dims[6] = { n_corr, n_T, nk, ns, ns, 2 };
        hid_t sp = H5Screate_simple(6, dims, nullptr);

        auto n_entries = n_corr * n_T * nk * ns * ns * 2;
        std::vector<double> buf, sq_buf;
        hid_t sum_ds=0, sum_sq_ds=0;


        buf.resize(n_entries);
        sum_ds = H5Dcreate2(grp, "static_corr", H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (store_error_term_) {
            sq_buf.resize(n_entries);
            sum_sq_ds = H5Dcreate2(grp, "static_corr_2", H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }

        if (sum_ds < 0 || sum_sq_ds < 0) {
            H5Sclose(sp);
            H5Gclose(grp);
            throw std::runtime_error("ssf_manager: failed to create static_corr or static_corr_2");
        }

        for (size_t c = 0; c < n_corr; ++c)
            for (size_t t = 0; t < n_T; ++t) {
                const auto& corr = corr_[c][ord[t]];
                const auto& corr_sq = corr_sq_[c][ord[t]];
                for (hsize_t mu = 0; mu < ns; ++mu)
                    for (hsize_t nu = 0; nu < ns; ++nu)
                        for (hsize_t k = 0; k < nk; ++k) {
                            const auto val = corr(static_cast<int>(mu),
                                                  static_cast<int>(nu))[k];
                            const auto sq_val = corr_sq(static_cast<int>(mu),
                                                  static_cast<int>(nu))[k];
                            const size_t base =
                                ((((c * n_T + t) * nk + k) * ns + mu) * ns + nu);
                            buf[base * 2]     = val.real();
                            buf[base * 2 + 1] = val.imag();

                            if (store_error_term_){
                                sq_buf[base * 2]     = sq_val.real();
                                sq_buf[base * 2 + 1] = sq_val.imag();
                            }
                        }
            }

        H5Dwrite(sum_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        H5Dclose(sum_ds);

        if (store_error_term_){
            H5Dwrite(sum_sq_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, sq_buf.data());
            H5Dclose(sum_sq_ds);
        }

        H5Sclose(sp);
    }

    // --- corr_lookup [n_corr] variable-length UTF-8 strings ---
    {
        hid_t str_t = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_t, H5T_VARIABLE);
        H5Tset_cset(str_t, H5T_CSET_UTF8);

        hsize_t nc = static_cast<hsize_t>(n_corr);
        hid_t sp = H5Screate_simple(1, &nc, nullptr);
        hid_t ds = H5Dcreate2(grp, "corr_lookup", str_t, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) {
            H5Sclose(sp);
            H5Tclose(str_t);
            H5Gclose(grp);
            throw std::runtime_error("ssf_manager: failed to create corr_lookup");
        }

        std::vector<const char*> ptrs(n_corr);
        for (size_t c = 0; c < n_corr; ++c)
            ptrs[c] = corr_specs_[c].label.c_str();
        H5Dwrite(ds, str_t, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data());
        H5Dclose(ds);
        H5Sclose(sp);
        H5Tclose(str_t);
    }

    // --- T_list, n_samples ---
    write_1d("T_list",    H5T_NATIVE_DOUBLE, n_T, sT.data());
    write_1d("n_samples", H5T_NATIVE_ULLONG, n_T, sN.data());

    // --- sl_positions [n_sl, 3] int64 ---
    // vec3<int64_t> has layout int64_t[3], so the vector is a contiguous int64 buffer.
    {
        hsize_t dims[2] = { ns, 3 };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, "sl_positions", H5T_NATIVE_INT64, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) {
            H5Sclose(sp);
            H5Gclose(grp);
            throw std::runtime_error("ssf_manager: failed to create sl_positions");
        }
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 sl_positions_.data());
        H5Dclose(ds);
        H5Sclose(sp);
    }

    // --- Attribute: n_spins ---
    {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t at = H5Acreate2(grp, "n_spins", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, &n_spins_);
        H5Aclose(at);
        H5Sclose(sp);
    }

    // --- Attribute: k_dims[3] ---
    {
        hsize_t adim = 3;
        int kd[3] = { static_cast<int>(k_dims_[0]),
                      static_cast<int>(k_dims_[1]),
                      static_cast<int>(k_dims_[2]) };
        hid_t sp = H5Screate_simple(1, &adim, nullptr);
        hid_t at = H5Acreate2(grp, "k_dims", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, kd);
        H5Aclose(at);
        H5Sclose(sp);
    }

    H5Gclose(grp);
}
