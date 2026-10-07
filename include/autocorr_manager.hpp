#pragma once

#include "H5Apublic.h"
#include "H5Dpublic.h"
#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Spublic.h"
#include "H5Tpublic.h"
#include "MC.hpp"
#include "abstract_manager.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <filesystem>
#include <fftw3.h>
#include <stdexcept>
#include <vector>


// Per-temperature autocorrelation function calculator.
//
// Collects a Monte-Carlo-time series of a single spin, chosen at construction
// time, for each temperature slot.  On write_group() the stored series are
// transformed in place into the (connected, normalised) spin autocorrelation
// function
//
//   A_i(tau) = C_i(tau) / C_i(0),
//   C_i(tau) = (1 / (N - tau)) * sum_{t=0}^{N-1-tau}
//                  (S_i(t) - <S_i>)(S_i(t + tau) - <S_i>)
//
// computed independently for each Cartesian component i = x, y, z.  A_i(0) == 1
// by construction; the decay of A_i(tau) measures how many sweeps are needed for
// the sampled spin to decorrelate.
//
// C_i(tau) is evaluated via the Wiener-Khinchin theorem — FFT, power spectrum,
// inverse FFT — in O(N log N) rather than the naive O(N^2).  The series is
// zero-padded to >= 2N before transforming so the circular FFT correlation
// reproduces the linear (non-wrapping) autocorrelation.
//
// HDF5 output layout (write_group, default group "/autocorr"):
//   autocorr   [n_T, n_tau]   — A_i(tau), double
//   T_list     [n_T]             — temperatures
//   n_samples  [n_T]             — series length N per temperature
//   attribute spin_ipos[3]       — integer position of the sampled spin
//   attribute pyro_sl            — its pyrochlore sublattice (0-3)
class autocorr_manager : public abstract_manager {
    using history_t = std::vector<double>;
    std::vector<history_t> S_history; // format: S_history[tau_slot][x,y,z][tau]

    const CMC::HeisenbergSpin* hs;
    bool finalised_ = false;

    void on_new_temp() override {
        S_history.push_back(history_t());
    }

    // Transforms S_history[idx] in place from a raw time series into the
    // per-component normalised connected autocorrelation function.
    void finalise(int idx);

    public:
    autocorr_manager(const CMC::HeisenbergSpin* _s, size_t n_temperatures_reserve=0) :
        hs(_s) {
        S_history.reserve(n_temperatures_reserve);
    }

    void sample(double dE){
        assert(!T_list.empty());
        assert(!finalised_ && "sample() called after the series were finalised");

        S_history[curr_idx].push_back(dE);

        n_samples[curr_idx]++;
    }

    void save(const std::filesystem::path& file_path);
    void write_group(hid_t file_id, const char* group_name="/autocorr") override;

};

inline void autocorr_manager::finalise(int idx){
    history_t& hist = S_history[idx];

    const size_t N = hist.size();
    if (N == 0)
        return;

    // Pad to the next power of two >= 2N so the circular FFT correlation equals
    // the linear autocorrelation (no wrap-around contamination).
    size_t M = 1;
    while (M < 2 * N) M <<= 1;

    const size_t n_freq = M / 2 + 1;
    double*        in   = fftw_alloc_real(M);
    double*        out  = fftw_alloc_real(M);
    fftw_complex*  freq = fftw_alloc_complex(n_freq);

    // Out-of-place plans with FFTW_ESTIMATE do not clobber their input buffers,
    // so both plans can be built once and reused across the three components.
    fftw_plan fwd = fftw_plan_dft_r2c_1d(static_cast<int>(M), in, freq, FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_c2r_1d(static_cast<int>(M), freq, out, FFTW_ESTIMATE);

    {
        // Connected series: subtract the mean, then zero-pad to length M.
        double mean = 0.0;
        for (double v : hist) mean += v;
        mean /= static_cast<double>(N);

        for (size_t t = 0; t < N; t++) in[t] = hist[t] - mean;
        std::fill(in + N, in + M, 0.0);

        fftw_execute(fwd);

        // Power spectrum (real, even) in place.
        for (size_t k = 0; k < n_freq; k++) {
            const double re = freq[k][0];
            const double im = freq[k][1];
            freq[k][0] = re * re + im * im;
            freq[k][1] = 0.0;
        }

        fftw_execute(bwd);

        // out[tau] = M * sum_{t} (x[t]-mean)(x[t+tau]-mean)  (c2r is unnormalised).
        std::vector<double> A(N);
        for (size_t tau = 0; tau < N; tau++)
            A[tau] = out[tau] / (static_cast<double>(M) * static_cast<double>(N - tau));

        // Normalise so A(0) == 1.  A(0) is the variance; if the component never
        // varied (variance 0) leave the autocorrelation identically zero.
        const double C0 = A[0];
        if (C0 > 0.0)
            for (double& a : A) a /= C0;
        else
            std::fill(A.begin(), A.end(), 0.0);

        hist = std::move(A);
    }

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(in);
    fftw_free(out);
    fftw_free(freq);
}

inline void autocorr_manager::save(const std::filesystem::path& file_path){

    // create HDF5 file
    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }
    write_group(file_id);

    // Close groups and file
    H5Fclose(file_id);
}

inline void autocorr_manager::write_group(hid_t file_id,
        const char* group_name) {

    // Transform every stored series into its autocorrelation function (once).
    if (!finalised_) {
        for (size_t t = 0; t < S_history.size(); t++)
            finalise(static_cast<int>(t));
        finalised_ = true;
    }

    hid_t data_group = H5Gcreate2(file_id, group_name,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (data_group < 0) {
        throw std::runtime_error("Failed to create group");
    }

    const size_t n_T = S_history.size();

    // Use the shortest series across temperatures so the on-disk array is
    // rectangular (in practice all temperatures share the same sample count).
    size_t n_tau = 0;
    for (size_t t = 0; t < n_T; t++) {
        const size_t len = S_history[t].size();
        if (t == 0 || len < n_tau) n_tau = len;
    }

    auto fail = [&](const std::string& msg) {
        H5Gclose(data_group);
        throw std::runtime_error(msg);
    };

    // autocorr[n_T, 3, n_tau], row-major contiguous buffer.
    {
        std::vector<double> buf(n_T * n_tau);
        for (size_t t = 0; t < n_T; t++)
            for (size_t tau = 0; tau < n_tau; tau++)
                buf[t  * n_tau + tau] = S_history[t][tau];

        const hsize_t dims[2] = { n_T, n_tau };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(data_group, "autocorr", H5T_NATIVE_DOUBLE, sp,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp); fail("Failed to create dataset autocorr"); }
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        H5Dclose(ds);
        H5Sclose(sp);
    }

    // 1-D metadata: T_list, n_samples.
    {
        const hsize_t len = n_T;
        auto write_1d = [&](const char* name, hid_t type, const void* data) {
            hid_t sp = H5Screate_simple(1, &len, nullptr);
            hid_t ds = H5Dcreate2(data_group, name, type, sp,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (ds < 0) { H5Sclose(sp);
                fail(std::string("Failed to create dataset ") + name); }
            H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            H5Dclose(ds);
            H5Sclose(sp);
        };
        write_1d("T_list",    H5T_NATIVE_DOUBLE, T_list.data());
        write_1d("n_samples", H5T_NATIVE_ULLONG, n_samples.data());
    }

    H5Gclose(data_group);
}
