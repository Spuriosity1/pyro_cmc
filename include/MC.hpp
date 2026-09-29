#pragma once
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Tpublic.h"
#include "vec3.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <map>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>

// LatticeLab 2
#include <lattice_lib/supercell.hpp>
#include <XoshiroCpp.hpp>
#include <random>


namespace CMC {

using json=nlohmann::json;
using dmat33_t = vector3::mat33<double>;


struct HeisenbergSpin;

/**
 * @brief A group of neighbours of one spin that all share the same coupling
 * matrix orientation.
 *
 * `J` points into the owning CouplingSpec (either its `J` or its transpose
 * `Jt`). The field contribution of this shell to its owner spin is
 * `*J * sum(bonds->S)`. A spin that is the *source* (lower pyro_sl) of a bond
 * points at `J`; the *target* (higher pyro_sl) points at `Jt`. A pointer (not a
 * reference) is used so BondShell stays copy-assignable, as build_supercell
 * requires of HeisenbergSpin. The pointee stays valid because `coupling_specs`
 * is never resized after setup_lattice().
 */
struct GenericBondShell {
    const dmat33_t* J;
    std::vector<HeisenbergSpin*> bonds;
};


/**
 * Like GenericBondShell, for the special (yet common) case that 
 * J is an identity matrix.
 */
struct HeisBondShell {
    const double* J;
    std::vector<HeisenbergSpin*> bonds;
};

// Biquadratic bonds, understood as K_{ij} (S_i \cdot S_j)^2
struct BQBondShell {
    const double* K;
    std::vector<HeisenbergSpin*> bonds;
};


/**
 * @brief Classical Heisenberg spin stored in the supercell.
 *
 * Satisfies the GeometricObject concept via the `ipos` member.
 *
 * `general_bonds` holds this spin's neighbours grouped by which oriented coupling
 * matrix applies (see BondShell); local_field sums J*neighbours over shells.
 * `heis_bonds` holds bilinear bonds that are pure Heisenberg.
 * `biquad_bonds` holds the bonds of the form K (S \cdot S)^2.
 */
struct HeisenbergSpin {
    ipos_t ipos;                           // required by GeometricObject concept
    int pyro_sl = 0;                       // pyrochlore sublattice 0–3
    vector3::vec3<double> S;
    int8_t lifted_dir = 1;                 // ±1; used by sweep_lifted_Metropolis, ignored otherwise

    std::vector<GenericBondShell> general_bonds;
    std::vector<HeisBondShell> heis_bonds;
    std::vector<BQBondShell> biquad_bonds;
};

/**
 * @brief Supercell of Heisenberg spins on the pyrochlore lattice.
 */
typedef Supercell<HeisenbergSpin> Lattice;


/**
 * @brief Specification of a single coupling term in the Hamiltonian.
 *
 * relative_vectors[sl] lists displacement vectors from a spin of pyrochlore
 * sublattice sl (0–3) to its coupled neighbors.
 */
struct GeneralCouplingSpec {
    std::string name;
    std::vector<std::vector<ipos_t>> relative_vectors;
    dmat33_t J;    // applied by the source (lower-pyro_sl) endpoint of a bond
    dmat33_t Jt;   // = J.tr(); applied by the target (higher-pyro_sl) endpoint
};

struct HeisenbergCouplingSpec {
    std::string name;
    std::vector<std::vector<ipos_t>> relative_vectors;
    double J;
};

struct BiquadraticCouplingSpec {
    std::string name;
    std::vector<std::vector<ipos_t>> relative_vectors;
    double K;
};

struct MC_parameters {
    double T_ref=1.0;
    size_t verbosity = 2;
    double hloc_atol = 1e-16;
    double u_atol = 1e-16;
};


struct Min_parameters {
    size_t verbosity = 2;
    double hloc_atol = 1e-16;
    double u_atol = 1e-16;
};


/**
 * @brief Base class containing the coupling layout.
 */
class InteractingHamiltonian {
    protected:
    std::vector<GeneralCouplingSpec> general_coupling_specs;
    std::vector<HeisenbergCouplingSpec> heis_coupling_specs;
    std::vector<BiquadraticCouplingSpec> biquad_coupling_specs;

    // these are currently only used to ensure that names are unique.
    using index_entry_t = std::pair<size_t, size_t>;
    std::map<std::string, index_entry_t> index; 

    vector3::vec3<double> global_field={0,0,0};


    Lattice* lat;
    XoshiroCpp::Xoroshiro128PlusPlus rng;

    // RNG generators for convenience
    std::normal_distribution<double> normal_dist;
    std::uniform_int_distribution<size_t> site_dist;
    std::exponential_distribution<double> exp_dist;
    std::uniform_real_distribution<double> rand01;

    // helper functions
    vector3::vec3d local_linear_field(const HeisenbergSpin* spin) const;
    vector3::vec3d local_field(const HeisenbergSpin* spin) const;

    // Single-site biquadratic energy at trial orientation S (K/2 convention).
    // Returns 0 when the spin has no biquad bonds, so pure-linear runs pay
    // nothing. Sums each bond touching the site once, so differences of this
    // quantity give the correct dE for a proposed move at that site.
    double biquad_site_energy(const HeisenbergSpin* spin,
            const vector3::vec3d& S) const;

    public:
    InteractingHamiltonian(Lattice& lat_, size_t seed) :
        lat(&lat_), rng(seed),
          site_dist(0, lat_.get_objects<HeisenbergSpin>().size()-1),
          rand01(0,1)
    {}

    // Repoints this runner at a different (but congruent) Lattice — O(1), no
    // spin data copied. Used by parallel tempering to swap which replica's
    // configuration a given temperature slot is currently simulating.
    void rebind(Lattice& new_lat);

    double total_energy_per_unit_cell() const;

    void define_general_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const vector3::mat33<double>& J
            );

    void define_Heisenberg_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const double J
            );

    void define_biquad_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const double K
            );

    void set_global_field(const vector3::vec3<double>& h);
    vector3::vec3d get_global_field() const;

    void setup_lattice();

    // Set every spin to an independent uniformly-random unit vector, so the
    // simulation never starts from the zero-initialised state (which leaves the
    // local field undefined for field-based moves). Overwritten by an explicit
    // initial state, e.g. init_spiral_state.
    void randomize_spins();
};

/**
 * @brief Monte Carlo driver for classical spin simulations.
 * The internal bond representation stores interactions in the form
 *
 *   \sum J_{ij}^{a b} S_i^a S_j^b 
 * + \sum Jh_{ij}      S_i^a S_j^a 
 * + \sum K_{ij}/2    (S_i^a S_j^a)^2
 *
 * Heisenberg interactions are stored separately for better performance.
 */
class MC_runner : public InteractingHamiltonian {

public:
    MC_parameters settings;

    MC_runner(Lattice &lat_, size_t seed)
        : InteractingHamiltonian(lat_, seed)
    {}

    size_t local_overrelax(double T, HeisenbergSpin* spin); //*
    size_t local_Metropolis(double T, HeisenbergSpin* spin); // ** 
    size_t local_lifted_Metropolis(double T, HeisenbergSpin* spin); // **
    size_t local_lifted_Metropolis_rot(double T, HeisenbergSpin* spin); // **

    size_t overrelax_all(double T);
    size_t sweep_local_Metropolis(double T, size_t n_overrelax=1);
    size_t sweep_lifted_Metropolis(double T, size_t n_overrelax=1);
    size_t sweep_lifted_Metropolis_rot(double T, size_t n_overrelax=1);
};



/**
 * @brief Outcome of a minimisation run.
 */
struct MinResult {
    int    iters      = 0;      // accepted steps (adaptive) or sweeps (align)
    int    fevals     = 0;      // gradient evaluations performed
    double E          = 0.0;    // final energy per unit cell
    double grad_max   = 0.0;    // final max tangential gradient norm
    double final_step = 0.0;    // last accepted step size (adaptive only)
    bool   converged  = false;  // hit the tolerance before max_iter
};

/**
 * @brief Zero-temperature energy minimiser for a Lattice of Heisenberg spins.
 *
 * The energy gradient at site i is `local_field(i) - global_field`; the physical
 * degrees of freedom are unit vectors, so all optimisers work with the
 * *tangential* (Riemannian) gradient g⊥ = g - (g·S) S and retract back to the
 * sphere by renormalising after each move.
 */
class Minimiser : public InteractingHamiltonian {

    // Scratch buffers reused across iterations to avoid per-step allocation.
    std::vector<vector3::vec3d> S0;   // configuration at the start of a trial step
    std::vector<vector3::vec3d> k1;   // -g⊥ at S0
    std::vector<vector3::vec3d> k2;   // -g⊥ at the Euler predictor
    std::vector<vector3::vec3d> Se;   // Euler predictor (low order)
    std::vector<vector3::vec3d> dS;   // legacy fixed-step buffer

    // Full energy gradient at site i, including the external field.
    vector3::vec3d grad(const HeisenbergSpin* spin) const {
        return local_field(spin) - global_field;
    }

    // Tangential (Riemannian) gradient: strip the radial part that a unit-length
    // constraint cannot act on. Returns g⊥ and reports its norm via `n`.
    static vector3::vec3d tangential(const vector3::vec3d& g,
                                     const vector3::vec3d& S, double& n) {
        vector3::vec3d gt = g - vector3::dot(g, S) * S;
        n = sqrt(vector3::dot(gt, gt));
        return gt;
    }

    // Retract p = S + h*d back onto the unit sphere. A vanishing result carries
    // no direction, so the spin is left untouched rather than yielding NaN.
    static vector3::vec3d retract(const vector3::vec3d& S,
                                  const vector3::vec3d& d, double h) {
        vector3::vec3d p = S + h * d;
        double n = sqrt(vector3::dot(p, p));
        return (n < 1e-14) ? S : p / n;
    }

    // Fill `dst` with -g⊥ for every spin at the current lattice configuration
    // and return the largest tangential gradient norm (the convergence measure).
    double descent_direction(const std::vector<HeisenbergSpin>& spins,
                             std::vector<vector3::vec3d>& dst) {
        double gmax = 0.0;
        dst.resize(spins.size());
        for (size_t i = 0; i < spins.size(); ++i) {
            double n;
            dst[i] = -tangential(grad(&spins[i]), spins[i].S, n);
            gmax = std::max(gmax, n);
        }
        return gmax;
    }

public:
    Min_parameters settings;

    Minimiser(Lattice& lat_, size_t seed) : InteractingHamiltonian(lat_, seed){}

    // -------- best-alignment (Gauss–Seidel / T=0 heat bath) ------------------
    // Point one spin antiparallel to its instantaneous local field, i.e. to the
    // exact single-site energy minimum. This is exact for bilinear + Zeeman
    // terms; the biquadratic contribution is linearised about the current S, so
    // for K≠0 it is a (still strongly descending) fixed-point iteration rather
    // than an exact solve. Returns 1 if the spin moved, 0 otherwise.
    size_t align_spin(HeisenbergSpin* spin); 

    // -------- fixed-step projected gradient descent (legacy) -----------------
    // Kept for reference / regression testing. Prefer minimise_adaptive().
    void gradient_descent(double step){
        auto& spins = lat->get_objects<HeisenbergSpin>();
        descent_direction(spins, dS);
        for (size_t i = 0; i < spins.size(); ++i)
            spins[i].S = retract(spins[i].S, dS[i], step);
    }

    /**
     * @brief Adds gaussian noise to spin config and renormalises
     */
    void fuzz_spins(double amplitude) {
        auto& spins = lat->get_objects<HeisenbergSpin>();
        for (size_t i = 0; i < spins.size(); ++i){
            auto& S = spins[i].S;
            S[0] += this->normal_dist(rng)*amplitude;
            S[1] += this->normal_dist(rng)*amplitude;
            S[2] += this->normal_dist(rng)*amplitude;

            double a = sqrt(dot(S,S));
            S = fabs(a) > 1e-12 ? S/a : vector3::vec3d{0,0,1};
        }
        
    }

    /**
     * @brief Adaptive-step projected gradient flow.
     *
     * Integrates the dissipative flow dS/dt = -g⊥ on the unit sphere with an
     * embedded Euler/Heun pair (orders 1 and 2). The Heun result is accepted;
     * their difference estimates the local error, which an I-controller keeps
     * within the mixed tolerance `atol + rtol·|S|`. The step dt therefore grows
     * on the smooth basin floor and shrinks through stiff/steep regions
     * automatically — no hand-tuned step_size.
     *
     * Terminates when the largest tangential gradient falls below
     * `atol + rtol·g0` (g0 = initial gradient) or after `max_iter` accepted
     * steps, whichever comes first.
     *
     * @param dt0  initial step guess (auto-corrected within a few iterations)
     */
    MinResult minimise_adaptive(int max_iter, double dt0,
                                double atol = 1e-8, double rtol = 1e-6);

    /**
     * @brief Best-alignment (Gauss–Seidel) sweeps.
     *
     * Sweeps the lattice repeatedly, snapping each spin to its exact single-site
     * minimum (see align_spin). Uses the freshly-updated neighbours within a
     * sweep, so it typically converges in far fewer sweeps than fixed-step
     * descent for bilinear models — this is the natural T=0 solver for Heisenberg
     * spins. Stops when the max tangential gradient drops below
     * `atol + rtol·g0`.
     */
    MinResult minimise_align(int max_iter,
                             double atol = 1e-8, double rtol = 1e-6); 


};



// Write a "/geometry" group to an open HDF5 file with lattice statistics
// that are fixed for this disorder realisation:
//
//   n_spins            (scalar)  — number of non-deleted spins
//   n_tetras_by_intact (hsize_t[5]) — n_tetras_by_intact[k] = number of
//                                     tetrahedra with exactly k intact spins
 void write_geometry_group(hid_t file_id, Lattice& sc,
                                 const char* group_name = "geometry");



void save_spin_state(Lattice& lat, const std::filesystem::path& file_path);
void save_ft_spin_state(Lattice& lat, const std::filesystem::path& file_path);

inline std::ostream& operator<<(std::ostream& os, const MinResult& mr) {
    os << "Minimisation results:" <<
        "\nConverged?\t"<< (mr.converged ? "yes" : "no") <<
        "\nE=        \t"<< mr.E <<
        "\nfevals=   \t"<< mr.fevals <<
        "\niters=    \t"<<mr.iters<<std::endl;
    return os;
}


}
