#pragma once
#include "MC.hpp"
#include "fftw3.h"
#include <complex>
#include <hdf5.h>

namespace CMC {

/**
 * @brief RAII wrapper for a 3D real-to-complex FFTW plan.
 *
 * This struct owns:
 *  - Real-space input buffer `tmp` of size n0*n1*n2
 *  - Reciprocal-space output buffer `tmp_k` of size n0*n1*(n2/2+1)
 *  - An FFTW plan performing an r2c transform
 *
 * Copying is disabled due to FFTW ownership rules.
 * Move operations are implemented via deep reallocation to ensure safety.
 */
struct FT_plan_3D {
    fftw_plan fft_plan = nullptr;
    fftw_complex* tmp_k = nullptr;
    double* tmp = nullptr;

    int n0 = 0, n1 = 0, n2 = 0;
    int n2k = 0;
    int nk = 0;

    FT_plan_3D(int n0_, int n1_, int n2_)
        : n0(n0_), n1(n1_), n2(n2_)
    {
        n2k = n2/2 + 1;
        nk = n0 * n1 * n2k;
        allocate_and_plan();
    }

    // --- internal helper ---
    void allocate_and_plan() {
        tmp = static_cast<double*>(
            fftw_malloc(sizeof(double) * (size_t)n0 * n1 * n2)
        );
        tmp_k = static_cast<fftw_complex*>(
            fftw_malloc(sizeof(fftw_complex) * (size_t)nk)
        );

        if (!tmp || !tmp_k)
            throw std::runtime_error("FFTW malloc failed");

        fft_plan = fftw_plan_dft_r2c_3d(n0, n1, n2, tmp, tmp_k, FFTW_ESTIMATE);
        if (!fft_plan)
            throw std::runtime_error("FFTW plan creation failed");
    }

    // --- no copy ---
    FT_plan_3D(const FT_plan_3D&) = delete;
    FT_plan_3D& operator=(const FT_plan_3D&) = delete;

    // --- move constructor ---
    FT_plan_3D(FT_plan_3D&& other) noexcept
        : fft_plan(nullptr),
          tmp_k(nullptr),
          tmp(nullptr),
          n0(other.n0),
          n1(other.n1),
          n2(other.n2),
          n2k(other.n2k),
          nk(other.nk)
    {
        // The safe version RE-ALLOCATES new buffers + plan
        allocate_and_plan();

        // Move contents (optional: zero out old)
        std::memcpy(tmp, other.tmp, sizeof(double) * (size_t)n0*n1*n2);
        std::memcpy(tmp_k, other.tmp_k, sizeof(fftw_complex) * (size_t)nk);

        // old becomes empty
        other.cleanup_after_move();
    }

    // --- move assignment ---
    FT_plan_3D& operator=(FT_plan_3D&& other) noexcept {
        if (this != &other) {
            cleanup_all();

            n0  = other.n0;
            n1  = other.n1;
            n2  = other.n2;
            n2k = other.n2k;
            nk  = other.nk;

            allocate_and_plan();

            std::memcpy(tmp, other.tmp,  sizeof(double) * (size_t)n0*n1*n2);
            std::memcpy(tmp_k, other.tmp_k, sizeof(fftw_complex)*(size_t)nk);

            other.cleanup_after_move();
        }
        return *this;
    }

    ~FT_plan_3D() {
        cleanup_all();
    }

    void transform() {
        fftw_execute(fft_plan);
    }

private:
    void cleanup_all() {
        if (fft_plan) fftw_destroy_plan(fft_plan);
        if (tmp_k)    fftw_free(tmp_k);
        if (tmp)      fftw_free(tmp);
        fft_plan = nullptr;
        tmp_k = nullptr;
        tmp = nullptr;
    }

    void cleanup_after_move() {
        fft_plan = nullptr;
        tmp_k = nullptr;
        tmp = nullptr;
        n0 = n1 = n2 = n2k = nk = 0;
    }
};


/* INDEXING BOOK-KEEPING
 *
 * latlib has very kindly made sure that the supercell has lattice vectors in the format
 *
 * [ A0 | A1 | A2  ] = [ a0 | a1 | a2 ] * diag(d0, d1, d2)
 * 
 * We now wish to compute the (un-normalised) FFT of some single-point function f(r), defined at lattice sites. 
 *
 * f(q) = \sum_{r} e^-i qT r f(r)
 *      =  \sum_sl exp(-i qT r_sl} * ( 
 *          \sum_{i0=0}^{d0-1} \sum_{i1=0}^{d1-1} \sum_{i2=0}^{d2-1}  {
 *              exp(-i qT [a] [i0; i1; i2]) f_{i0,i1,i2; sl}
 *              }
 *          )
 *  The ft_plan objects compute the intra-sl correlations {the object in braces},
 *  storing in the sl-local tmp_k.
 *          
 *  We also make allowance for a sl-dependent local frame, implemented as a g-tensor
 *
 *  m^a_{r; sl} = g^ab_{sl} S^b_{r; sl}
 *
 *  To calculate <m^a(q) m^b(-q)> === <m^a(q) conj[ m^b(q) ]>, we evaluate
 *
 *  F_{sl}^b(h0, h1, h2) = FFT( f_{i0,i1,i2; sl}, h0, h1, h2)
 *
 *  =  \sum_{i0=0}^{d0-1} \sum_{i1=0}^{d1-1} \sum_{i2=0}^{d2-1}  {
 *              exp(-2 pi i [h0 h1 h2] [i0; i1; i2]) S^b_{i0,i1,i2; sl}
 *              }
 *
 *  where the col vector of integers [h] is understood as 
 *  2 pi h = aT q
 *
 *  The magnetic structure factor <m^a1(q) m^a2(q)> is then
 *
 *  \sum_{sl1 b1; sl2 b2 } g_{sl1}^{a1 b1} g_{sl2}^{a2 b2} 
 *          exp[-i qT (r_sl1 - r_sl2)] 
 *              F_{sl1}^b1 (1/2pi aT q) conj[ F_{sl2}^b2( 1/2pi aT q ) ]
 *
 *
 *
 */



/**
 * @brief Container for a single correlation observable in momentum space.
 */
struct corr_fn {
    // Human readable name (e.g. "Szz")
    std::string name;
    // Bitmask specifying which correlators to compute
    uint16_t mask;
    // Correlator data accumulator (size = # k-points)
    std::vector<double> data;

    corr_fn(const std::string &name_, uint16_t mask_, size_t n_sites_)
        : name(name_), mask(mask_), data(n_sites_) {}
};





/**
 * @brief Static (equal-time) spin–spin correlations in 3D momentum space.
 *
 * Computes and accumulates structure factors using FFTs over the full
 * periodic supercell, resolving sublattice and spin components.
 */
class static_corr_3D {
    const Lattice& lat;
    std::vector<vector3::vec3d> sublattice_positions;

    const vector3::mat33<double> recip_latvecs;

    inline sl_t n_sublattices() const {
      return this->sublattice_positions.size();
    }

    inline sl_t n_k_points() const {
        // roughly half of n_sublattices
        const int n0 = lat.size(0);
        const int n1 = lat.size(1);
        const int n2 = lat.size(2);
        const int n2k = n2 / 2 + 1;
        return n0 * n1 * n2k;
    }

    uint16_t needed_pairs=0; // bitmask
    // the point:
    
    uint16_t compute_correlators=0; // bitmask stores which observables to compute
    std::vector<corr_fn> observables;  

    // first index is the sublattice
    std::vector<FT_plan_3D> ft_plans_X;
    std::vector<FT_plan_3D> ft_plans_Y; 
    std::vector<FT_plan_3D> ft_plans_Z; 
                                      
	int n_samples=0; // number of sampels stored in SQQ
                
    
    // Evaluates sum_{q} conj(f1.tmp_k(q)) * f2.tmp_k(q)
    // note: tmp_k is stored in shape (n0, n1, n2k)
    void k_inner(const std::vector<FT_plan_3D>& fft_vec_a, sl_t sl_a,
            const std::vector<FT_plan_3D>& fft_vec_b, sl_t sl_b,
            std::vector<double>& res) const{
        const int n0 = lat.size(0);
        const int n1 = lat.size(1);
//        const int n2 = lat.size(2);
//
        const auto& fft_a = fft_vec_a[sl_a];
        const auto& fft_b = fft_vec_b[sl_b];

        // complex last-dim length used by r2c
        const int n2k = fft_a.n2k; // should equal fft_b.n2k

        // sanity check: ensure res has the expected size
        const size_t expected = static_cast<size_t>(n0) * n1 * n2k;
        if (res.size() != expected) {
            throw std::runtime_error("k_inner: res vector size mismatch ("+
                    std::to_string(res.size()) + ") expected "+
                    std::to_string(expected));
        }

        const vector3::vec3d dr = sublattice_positions[sl_a] - sublattice_positions[sl_b];

        for (int i0 = 0; i0 < n0; ++i0) {
            for (int i1 = 0; i1 < n1; ++i1) {
                for (int i2 = 0; i2 < n2k; ++i2) {
                    const auto I = i2 + n2k * (i1 + n1 * i0);
                    const auto& m_a = fft_a.tmp_k[I];
                    const auto& m_b = fft_b.tmp_k[I];

                    double phase = vector3::dot(dr, recip_latvecs * 
                            vector3::vec3d(i0, i1, i2));

                    // conj(m_a) * m_b = (a_re - i a_im)*(b_re + i b_im) = a_re*b_re + a_im*b_im + i(...)
                    // we store the real part (power) a_re*b_re + a_im*b_im
                    res[I] += (m_a[0] * m_b[0] + m_a[1] * m_b[1]) * cos(phase)
                        + (m_a[0] * m_b[1] - m_a[1] * m_b[0]) * sin(phase);
                }
            }
        }
    }


public:
    static_corr_3D(const Lattice& lat_) :
        lat(lat_), 
        recip_latvecs(vector3::mat33<double>::from_other(
                    get_reciprocal_cell_vectors(lat.index_cell_vectors)
                    ))
    {
        for (int i=0; i<lat.primitive_spec.num_link_sl(); i++){
            sublattice_positions.push_back(lat.primitive_spec.link_no(i).position);
        }

        auto slcount=n_sublattices();
        ft_plans_X.reserve(slcount);
        ft_plans_Y.reserve(slcount);
        ft_plans_Z.reserve(slcount);

        for (sl_t sl=0; sl<n_sublattices(); sl++){
            ft_plans_X.emplace_back(lat.size(0), lat.size(1), lat.size(2));
            ft_plans_Y.emplace_back(lat.size(0), lat.size(1), lat.size(2));
            ft_plans_Z.emplace_back(lat.size(0), lat.size(1), lat.size(2));
        }


    }
    ~static_corr_3D(){
//        fftw_free(SQQ);
    }


    // We don't want to waste time computing FT's we don't need.
    // If all observableds are diagonal, we just calculate this once:
    static const uint16_t NEEDS_XX = 0x0001;
    static const uint16_t NEEDS_YY = 0x0002;
    static const uint16_t NEEDS_ZZ = 0x0004;
    static const uint16_t NEEDS_XY = 0x0010;
    static const uint16_t NEEDS_ZX = 0x0020;
    static const uint16_t NEEDS_ZY = 0x0030;


    constexpr static std::array<uint16_t, 9> NEEDS = {
        NEEDS_XX, NEEDS_YY, NEEDS_ZZ,
        NEEDS_ZX, NEEDS_ZY, NEEDS_XY
    };


    void declare_observable(const std::string& name, uint16_t correlators_to_count ){
        if (correlators_to_count == 0) {
            throw std::logic_error("Must request at least one correlator when declaring an observable");
        }
        observables.emplace_back(name, correlators_to_count, n_k_points());
        compute_correlators |= correlators_to_count;
    }

    void sample();

    void write_group(hid_t file_id, const char* group_name="/ssf");
    void save(const std::filesystem::path& file_path);

    void write_K_points(hid_t file_id, const char* dset_name="k_points");
    void save_K_points(const std::filesystem::path& file_path);


};




} // namespace CMC
