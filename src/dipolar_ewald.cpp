#include "dipolar_ewald.hpp"
#include <cmath>

namespace CMC {

namespace {

// Cartesian column i of an integer lattice-vector matrix, scaled to physical
// units (ipos / pos_scale).
inline vector3::vec3d col_cartesian(const imat33_t& M, int i, double scale) {
    return vector3::vec3d(static_cast<double>(M(0, i)),
                          static_cast<double>(M(1, i)),
                          static_cast<double>(M(2, i))) *= (1.0 / scale);
}

inline double col_norm(const dmat33_t& M, int i) {
    vector3::vec3d c = M.col(i);
    return std::sqrt(dot(c, c));
}

} // namespace

DipolarEwald::DipolarEwald(const LatticeIndexing& lattice,
                           const std::vector<ipos_t>& sl_pos,
                           double D,
                           EwaldParams params)
    : lattice_(&lattice),
      n_sl_(static_cast<int>(sl_pos.size())),
      Np_(lattice.num_primitive_cells())
{
    const double scale = params.pos_scale;

    // --- Cartesian supercell geometry -------------------------------------
    // Direct lattice vectors (columns) of the periodic supercell, in physical
    // units, and the matching reciprocal vectors (carrying the 2π factor).
    const imat33_t latvec_i = lattice.get_lattice_vectors();
    dmat33_t Lc;   // columns: supercell direct lattice vectors (Cartesian)
    dmat33_t Bc;   // columns: supercell reciprocal lattice vectors
    {
        const dmat33_t recip = lattice.get_reciprocal_lattice_vectors();
        for (int i = 0; i < 3; ++i) {
            vector3::vec3d a = col_cartesian(latvec_i, i, scale);
            // get_reciprocal_lattice_vectors() is the reciprocal of the integer
            // lattice; the Cartesian (scaled) lattice is 1/scale as large, so its
            // reciprocal is scale× larger.
            vector3::vec3d b = scale * recip.col(i);
            for (int r = 0; r < 3; ++r) { Lc(r, i) = a[r]; Bc(r, i) = b[r]; }
        }
    }

    // Primitive-cell translation per cell index: PrimC·Δidx3 (Cartesian).
    // index_cell = prim · diag(size), so prim column i = Lc column i / size_i.
    const idx3_t sc_size = lattice.size();
    dmat33_t PrimC;
    for (int i = 0; i < 3; ++i)
        for (int r = 0; r < 3; ++r)
            PrimC(r, i) = Lc(r, i) / static_cast<double>(sc_size[i]);

    const double V = std::abs(det(Lc));
    double box_min = col_norm(Lc, 0), box_max = col_norm(Lc, 0);
    for (int i = 1; i < 3; ++i) {
        box_min = std::min(box_min, col_norm(Lc, i));
        box_max = std::max(box_max, col_norm(Lc, i));
    }

    // --- Ewald parameters / truncation ------------------------------------
    double alpha = params.alpha;
    if (alpha <= 0.0) alpha = 3.2 / box_min;

    const double xcut = std::sqrt(-std::log(params.tol)); // e^{-x²} < tol
    const double sqrt_pi = std::sqrt(M_PI);

    // Real-space: include images until erfc(α r) ≲ tol, allowing for intra-box
    // displacements up to one full box extent.
    const double r_max = xcut / alpha;
    const int n_real = static_cast<int>(
        std::ceil((r_max + box_max) / box_min)) + 1;

    // Reciprocal-space: |k| up to where e^{-k²/4α²} ≲ tol.
    const double k_max = 2.0 * alpha * xcut;
    const double dk_min = 2.0 * M_PI / box_max;
    const int n_k = static_cast<int>(std::ceil(k_max / dk_min)) + 1;
    const double gauss_k = 1.0 / (4.0 * alpha * alpha);

    const double self_diag = -4.0 * alpha * alpha * alpha / (3.0 * sqrt_pi);

    // --- Build the tensor table -------------------------------------------
    A_.assign(static_cast<size_t>(n_sl_) * n_sl_ * Np_, dmat33_t{});

    // Cartesian sublattice positions.
    std::vector<vector3::vec3d> rsl(n_sl_);
    for (int s = 0; s < n_sl_; ++s)
        rsl[s] = vector3::vec3d(static_cast<double>(sl_pos[s][0]),
                                static_cast<double>(sl_pos[s][1]),
                                static_cast<double>(sl_pos[s][2])) *= (1.0 / scale);

    for (int a = 0; a < n_sl_; ++a) {
        for (int b = 0; b < n_sl_; ++b) {
            for (idx_t dflat = 0; dflat < Np_; ++dflat) {
                const idx3_t Didx = lattice.idx3_from_flat(dflat);

                // Base displacement r_b − r_a for this (sublattice pair, cell Δ).
                vector3::vec3d d0 = rsl[b] - rsl[a];
                for (int i = 0; i < 3; ++i)
                    d0 += static_cast<double>(Didx[i]) * PrimC.col(i);

                double acc[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

                // Real-space screened sum over periodic images.
                for (int n0 = -n_real; n0 <= n_real; ++n0)
                for (int n1 = -n_real; n1 <= n_real; ++n1)
                for (int n2 = -n_real; n2 <= n_real; ++n2) {
                    vector3::vec3d r = d0;
                    r += static_cast<double>(n0) * Lc.col(0);
                    r += static_cast<double>(n1) * Lc.col(1);
                    r += static_cast<double>(n2) * Lc.col(2);
                    const double r2 = dot(r, r);
                    if (r2 < 1e-18) continue; // the self image (handled below)
                    const double rr = std::sqrt(r2);
                    if (rr > r_max) continue;
                    const double ar = alpha * rr;
                    const double erfc_t = std::erfc(ar);
                    const double gauss = std::exp(-ar * ar);
                    const double two_a = 2.0 * alpha / sqrt_pi;
                    // A_real = B δ − C r⊗r,  A = −∇∇(erfc(αr)/r)
                    const double B = erfc_t / (r2 * rr) + two_a * gauss / r2;
                    const double C = 3.0 * erfc_t / (r2 * r2 * rr)
                                   + two_a * (2.0 * alpha * alpha + 3.0 / r2) * gauss / r2;
                    for (int p = 0; p < 3; ++p) {
                        for (int q = 0; q < 3; ++q)
                            acc[p][q] -= C * r[p] * r[q];
                        acc[p][p] += B;
                    }
                }

                // Reciprocal-space sum (tinfoil: skip k = 0).
                for (int m0 = -n_k; m0 <= n_k; ++m0)
                for (int m1 = -n_k; m1 <= n_k; ++m1)
                for (int m2 = -n_k; m2 <= n_k; ++m2) {
                    if (m0 == 0 && m1 == 0 && m2 == 0) continue;
                    vector3::vec3d k{0,0,0};
                    k += static_cast<double>(m0) * Bc.col(0);
                    k += static_cast<double>(m1) * Bc.col(1);
                    k += static_cast<double>(m2) * Bc.col(2);
                    const double k2 = dot(k, k);
                    const double env = std::exp(-k2 * gauss_k);
                    if (env < params.tol) continue;
                    const double pref = (4.0 * M_PI / V) * env / k2 * std::cos(dot(k, d0));
                    for (int p = 0; p < 3; ++p)
                        for (int q = 0; q < 3; ++q)
                            acc[p][q] += pref * k[p] * k[q];
                }

                // On-site self correction.
                if (a == b && dflat == 0)
                    for (int p = 0; p < 3; ++p) acc[p][p] += self_diag;

                dmat33_t block{};
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q)
                        block(p, q) = D * acc[p][q];
                A_[(static_cast<size_t>(a) * n_sl_ + b) * Np_ + dflat] = block;
            }
        }
    }
}

} // namespace CMC
