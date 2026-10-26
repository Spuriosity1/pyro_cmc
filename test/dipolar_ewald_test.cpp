// dipolar_ewald_test.cpp
//
// Validates the long-range dipolar interaction (DipolarEwald + the cached field
// wired into MC_runner). With no exchange couplings the Hamiltonian is purely
// dipolar, so total_energy_per_unit_cell() measures only the dipolar energy.
//
// Checks, on a small pyrochlore supercell:
//   1. Cache vs direct      — the cache-based total energy equals the O(N²)
//                             double sum straight from the Ewald tensor.
//   2. alpha-independence    — the Ewald sum is independent of the (arbitrary)
//                             splitting parameter: real + reciprocal + self and
//                             the tinfoil convention are mutually consistent.
//   3. Incremental drift     — after many single-spin updates the incrementally
//                             maintained field cache still matches a fresh
//                             rebuild (apply_dipolar_update is exact).

#include <cmath>
#include <cstdio>
#include <cstdint>

#include <lattice_lib/supercell.hpp>
#include "MC.hpp"
#include "pyrochlore_geometry.hpp"

using namespace CMC;

namespace {

Lattice make_lat(int L) {
    auto cell = PyroCubicCell();
    auto supercell_spec = imat33_t::from_cols({L, 0, 0}, {0, L, 0}, {0, 0, L});
    return build_supercell(cell, supercell_spec);
}

bool close(double a, double b, double rtol, double atol) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

int fail(const char* name, double got, double want, double diff) {
    std::printf("  FAIL %s: got %.12g want %.12g (|diff|=%.3g)\n",
                name, got, want, diff);
    return 1;
}

} // namespace

int main() {
    const int L = 2;              // 16 * L^3 = 128 spins
    const double D = 1.0;
    const uint64_t seed = 12345;

    auto lat = make_lat(L);
    const double Np = lat.lattice.num_primitive_cells();

    int failures = 0;

    // ---- 1. cache vs direct ---------------------------------------------
    {
        MC_runner mc(lat, seed);
        mc.setup_lattice();            // no couplings => purely dipolar
        mc.enable_dipolar(D);
        mc.randomize_spins();          // also rebuilds the field cache

        const double e_cache  = mc.total_energy_per_unit_cell() * Np;
        const double e_direct = mc.total_dipolar_energy_direct();
        const double d = std::fabs(e_cache - e_direct);
        std::printf("[1] cache=%.10g  direct=%.10g  |diff|=%.3g\n",
                    e_cache, e_direct, d);
        if (!close(e_cache, e_direct, 1e-10, 1e-9))
            failures += fail("cache-vs-direct", e_cache, e_direct, d);
    }

    // ---- 2. alpha-independence ------------------------------------------
    {
        MC_runner mc(lat, seed);
        mc.setup_lattice();
        mc.enable_dipolar(D);
        mc.randomize_spins();          // fix a configuration

        EwaldParams p1; p1.alpha = 2.0 / L;
        EwaldParams p2; p2.alpha = 5.0 / L;
        mc.enable_dipolar(D, p1);      // rebuild tensor+cache, spins unchanged
        const double e1 = mc.total_dipolar_energy_direct();
        mc.enable_dipolar(D, p2);
        const double e2 = mc.total_dipolar_energy_direct();
        const double d = std::fabs(e1 - e2);
        std::printf("[2] E(alpha=%.3g)=%.10g  E(alpha=%.3g)=%.10g  |diff|=%.3g\n",
                    p1.alpha, e1, p2.alpha, e2, d);
        if (!close(e1, e2, 1e-7, 1e-8))
            failures += fail("alpha-independence", e1, e2, d);
    }

    // ---- 3. incremental drift -------------------------------------------
    {
        MC_runner mc(lat, seed);
        mc.setup_lattice();
        mc.enable_dipolar(D);
        mc.randomize_spins();

        const double T = 0.5;
        for (int s = 0; s < 2000; ++s)
            mc.sweep_local_Metropolis(T, /*n_overrelax=*/1);

        const double e_inc = mc.total_energy_per_unit_cell() * Np;  // uses the cache
        mc.rebuild_dipolar_field();                                 // flush round-off
        const double e_reb = mc.total_energy_per_unit_cell() * Np;
        const double d = std::fabs(e_inc - e_reb);
        std::printf("[3] incremental=%.10g  rebuilt=%.10g  |diff|=%.3g\n",
                    e_inc, e_reb, d);
        // ~2000 sweeps * 128 spins of O(1e-16) round-off; allow a generous bound.
        if (!close(e_inc, e_reb, 1e-8, 1e-7))
            failures += fail("incremental-drift", e_inc, e_reb, d);
    }

    if (failures == 0) {
        std::printf("dipolar_ewald_test: all checks passed\n");
        return 0;
    }
    std::printf("dipolar_ewald_test: %d check(s) failed\n", failures);
    return 1;
}
