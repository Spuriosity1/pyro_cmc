#pragma once
#include <lattice_lib/supercell.hpp>
#include <vector>

namespace CMC {

/**
 * @brief Parameters controlling the one-time Ewald lattice sum.
 *
 * `alpha` is the Ewald splitting parameter (1/length, in the same Cartesian
 * units as `pos_scale`). A non-positive value requests an automatic choice
 * (~3.2 / shortest supercell edge), which balances the real- and
 * reciprocal-space work and converges both to `tol`.
 *
 * `tol` sets where the real- and reciprocal-space sums are truncated: shells
 * are included until their Gaussian/erfc envelope drops below `tol`.
 *
 * `pos_scale` converts integer lattice coordinates to Cartesian: on the
 * pyrochlore lattice the conventional cubic cell is 8 integer units, so
 * `r_cartesian = ipos / 8`.
 */
struct EwaldParams {
    double alpha = -1.0;
    double tol = 1e-15;
    double pos_scale = 8.0;
};

/**
 * @brief Precomputed, translationally-invariant dipolar coupling tensor.
 *
 * Because spins sit on fixed sites and only rotate, the conditionally
 * convergent dipolar lattice sum under periodic boundary conditions collapses
 * to a static 3x3 tensor that depends only on the sublattice pair (a,b) and the
 * primitive-cell displacement Δ. This class evaluates that Ewald sum once at
 * construction (real-space erfc-screened part + reciprocal-space part + the
 * on-site self term), using tinfoil (metallic) boundary conditions — the k=0
 * depolarizing term is dropped.
 *
 * The stored energy convention is
 *   E_dip = ½ Σ_{i,j} S_i · A(a_i,a_j,R_i−R_j) · S_j ,
 * with A(r) → D (δ_{αβ}/r³ − 3 r_α r_β/r⁵) in the bare (single-image) limit.
 * `A` is symmetric under A(a,b,Δ) = A(b,a,−Δ)^T.
 */
class DipolarEwald {
public:
    DipolarEwald(const LatticeIndexing& lattice,
                 const std::vector<ipos_t>& sl_pos,
                 double D,
                 EwaldParams params = {});

    int num_sl() const { return n_sl_; }
    idx_t num_cells() const { return Np_; }

    // Coupling tensor from a site of sublattice b at cell-displacement
    // Δ (= cell_b − cell_a) to a site of sublattice a at the origin cell:
    //   g_a = Σ_b Σ_Δ block(a, b, Δ) · S_{b,Δ}.
    const dmat33_t& block(int sl_a, int sl_b, idx_t dflat) const {
        return A_[(static_cast<size_t>(sl_a) * n_sl_ + sl_b) * Np_ + dflat];
    }

    // On-site block A_ii = block(sl, sl, 0), carrying the lattice self term.
    // Used for the orientation-dependent dipolar self-energy ½ S·A_ii·S.
    const dmat33_t& self_block(int sl) const {
        return A_[(static_cast<size_t>(sl) * n_sl_ + sl) * Np_ + 0];
    }

    // Flat cell-displacement index for Δ = cell_b − cell_a, wrapped into the
    // supercell. flat_from_idx3_wrapped({0,0,0}) == 0.
    idx_t disp_flat(const idx3_t& cell_a, const idx3_t& cell_b) const {
        return lattice_->flat_from_idx3_wrapped(cell_b - cell_a);
    }

private:
    const LatticeIndexing* lattice_;
    int n_sl_;
    idx_t Np_;
    std::vector<dmat33_t> A_;   // flat [n_sl][n_sl][Np]
};

} // namespace CMC
