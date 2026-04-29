#pragma once
#include "MC.hpp"
#include <lattice_lib/fourier.hpp>
#include <complex>
#include <hdf5.h>
#include <filesystem>

namespace CMC {

// Component accessors for FourierTransformC2C
namespace spin_access {
    struct X { static double get(const HeisenbergSpin& s) { return s.S[0]; } };
    struct Y { static double get(const HeisenbergSpin& s) { return s.S[1]; } };
    struct Z { static double get(const HeisenbergSpin& s) { return s.S[2]; } };
}

using FT_Sx = FourierTransformC2C<HeisenbergSpin,
              FieldAccessor<HeisenbergSpin, &spin_access::X::get>,
              HeisenbergSpin>;
using FT_Sy = FourierTransformC2C<HeisenbergSpin,
              FieldAccessor<HeisenbergSpin, &spin_access::Y::get>,
              HeisenbergSpin>;
using FT_Sz = FourierTransformC2C<HeisenbergSpin,
              FieldAccessor<HeisenbergSpin, &spin_access::Z::get>,
              HeisenbergSpin>;


struct corr_fn {
    std::string name;
    uint16_t mask;
    std::vector<double> data;   // accumulated real part of contracted S(q), size n0*n1*n2

    corr_fn(const std::string& name_, uint16_t mask_, size_t n_kpts)
        : name(name_), mask(mask_), data(n_kpts, 0.0) {}
};


/**
 * @brief Static (equal-time) spin–spin correlations in 3D momentum space.
 *
 * Uses latlib2 FourierTransformC2C (c2c) for sublattice-resolved FFTs.
 * k-space dimensions are (n0, n1, n2) — full c2c, not r2c half-space.
 *
 * Phase convention: see SublatWeightMatrix::phase_factors in fourier.hpp.
 */
class static_corr_3D {
    Lattice& lat;

    FT_Sx ft_x;
    FT_Sy ft_y;
    FT_Sz ft_z;

    SublatWeightMatrix phase_mat;

    uint16_t compute_correlators = 0;
    std::vector<corr_fn> observables;
    int n_samples = 0;

    inline int n_k_points() const {
        return lat.lattice.size(0) * lat.lattice.size(1) * lat.lattice.size(2);
    }

    static SublatWeightMatrix make_phase_mat(Lattice& lat);

public:
    static const uint16_t NEEDS_XX = 0x0001;
    static const uint16_t NEEDS_YY = 0x0002;
    static const uint16_t NEEDS_ZZ = 0x0004;
    static const uint16_t NEEDS_XY = 0x0010;
    static const uint16_t NEEDS_ZX = 0x0020;
    static const uint16_t NEEDS_ZY = 0x0030;

    explicit static_corr_3D(Lattice& lat_);

    void declare_observable(const std::string& name, uint16_t correlators_to_count);

    void sample();

    void write_group(hid_t file_id, const char* group_name = "/ssf");
    void save(const std::filesystem::path& file_path);

    void write_K_points(hid_t file_id, const char* dset_name = "k_points");
    void save_K_points(const std::filesystem::path& file_path);
};


} // namespace CMC
