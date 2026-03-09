#pragma once
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Tpublic.h"
#include <nlohmann/json.hpp>

// LatticeLab
#include <chain.hpp>
#include <lattice_IO.hpp>
#include <cell_geometry.hpp>
#include <preset_cellspecs.hpp>
#include <UnitCellSpecifier.hpp>
#include <XoshiroCpp.hpp>
#include <random>


namespace CMC {

using json=nlohmann::json;
using namespace CellGeometry;



/**
 * @brief Zero-dimensional cell representing a tetrahedron.
 *
 * This is primarily a topological/organizational object used by the lattice
 * geometry. It carries no additional data beyond its Cell<0> identity.
 */
struct Tetra : public Cell<0> {
};

struct HeisenbergSpin;


/**
 * @brief Collection of neighboring spins connected by a given bond shell.
 *
 * For a fixed reference spin and a fixed coupling shell (e.g. nearest neighbors,
 * next-nearest neighbors, etc.), neighbors are split into two groups depending
 * on bond orientation.
 *
 * This separation is required to correctly implement antisymmetric interactions
 * such as Dzyaloshinskii–Moriya (DM) couplings, where bond orientation matters.
 */
struct NeighbourSpins {
    std::vector<HeisenbergSpin*> bonds_above;
    std::vector<HeisenbergSpin*> bonds_below;
};



/**
 * @brief Classical Heisenberg spin living on a lattice link (Cell<1>).
 *
 * Each spin:
 *  - Stores its current classical spin vector S
 *  - Maintains neighbor lists grouped by coupling shell
 *
 * Bond orientation convention:
 *  - If antisymmetric interactions (e.g. DM terms) are present, the direction
 *    of a bond must be tracked.
 *  - For a given bond between spins i and j:
 *      * If j <= i (pointer comparison), the bond is stored in bonds_above
 *      * If j >  i, the bond is stored in bonds_below
 *
 * This ensures that each bond has a well-defined orientation without duplicating
 * storage or introducing explicit bond objects.
 */
struct HeisenbergSpin : public Cell<1> {
    vector3::vec3<double> S;

    /**
     * Neighbor lists grouped by coupling shell.
     *
     * bond_sets[n] corresponds to the nth defined coupling (e.g. nn, nnn, etc.).
     * Each entry contains two lists distinguishing bond orientation.
     */
    std::vector<NeighbourSpins> bond_sets;
};


// Higher-dimensional cells (plaquettes, volumes) are currently unused but
// kept here as a reminder of possible future extensions.
//
// struct Plaq : public Cell<2> {
//     const Plaq* root = nullptr;
// };
//
// struct Vol : public Cell<3> {
//     const Vol* root = nullptr;
// };

/**
 * @brief Periodic lattice of tetrahedra with Heisenberg spins on links.
 *
 * The lattice manages all geometric connectivity and periodic boundary
 * conditions. Physical interactions are defined separately in MC_runner.
 */
typedef PeriodicLinkLattice<Tetra, HeisenbergSpin> Lattice;


/**
 * @brief Specification of a single coupling term in the Hamiltonian.
 *
 * Each coupling is defined by:
 *  - A human-readable name (e.g. "J1", "J2")
 *  - A set of relative lattice vectors defining which neighbors are coupled
 *  - A 3×3 interaction matrix J^{ab}
 */
struct CouplingSpec {
    std::string name;
    std::vector<std::vector<idx3_t>> relative_vectors;
    vector3::mat33<double> J;
};

//enum class UpdateStrategy { sequential, random};

// Global Monte Carlo parameters.
struct MC_parameters {
    /**reference T for the Gaussian update size heuristic. 
     * It should be comparable with the energy scales in e.g. the coupling matrices.
     */
    double T_ref=1.0; 
    
    // for diagnostics and logging, no physical consequence
    size_t verbosity = 2;
};

/**
 * @brief Monte Carlo driver for classical spin simulations.
 *
 * a "thin" wrapper around Lattice. It
 * - contains information about the coupling strengths 
 *   (actual connectivity managed by Lattice)
 * - computes local fields and energies
 * - performs Metropolis updates
 * 
 * The model: 
 * $$
 *  H = 1/2 \sum_{i \in Spins}  S_i^a {
 *              \sum_{n =1}^ncouplings J_(n)^{ab} (
 *                  \sum_{j -- coupling_set[n]} S_j^i
 *              )
 *      }
 * $$
**/
class MC_runner {
    // list of all the coupling specifications
    std::vector<CouplingSpec> coupling_specs;
    // map from coupling name to index in coupling_specs
    std::map<std::string, size_t> index;

    vector3::vec3<double> global_field={0,0,0};

    // reference to the underlying lattice (not owned)
    Lattice& lat;


    // -- RNG --
    std::uniform_int_distribution<size_t> site_dist;
    std::normal_distribution<double> normal_dist;
    std::uniform_real_distribution<double> rand01;

    // the underlying engine (can be swapped out for e.g. std::mt1337)
    XoshiroCpp::Xoroshiro128PlusPlus rng;


    // helper funciton evaluating local field at a spin site
    vector3::vec3d local_field(const HeisenbergSpin* spin) const;

public:
    MC_parameters settings;

    MC_runner(Lattice &lat_, size_t seed)
        : lat(lat_), site_dist(0, lat.links.size()), rand01(0,1), rng(seed)
    {
    }

    double total_energy_per_unit_cell() const;


    /**
     * @brief Define a coupling term in the Hamiltonian.
     *
     * @param name      Identifier for the coupling
     * @param rel_vecs  Relative displacement vectors defining the coupled neighbors
     * @param J         3×3 interaction tensor
     *
     * This function only registers the coupling; neighbor pointers are
     * constructed later by setup_lattice().
     */
    void define_coupling(const std::string& name, 
            const std::vector<std::vector<idx3_t>>& rel_vecs,
            const vector3::mat33<double>& J);


    /**
     * @brief Define a one-body field term in the Hamiltonian.
     *
     * @param h         3-vector local field, global coordinates
     *
     * This function only registers the coupling; neighbor pointers are
     * constructed later by setup_lattice().
     *
     * Different fields for diifferent sublattices currently not implemented.
     */
    void define_global_field(const vector3::vec3<double>& h);

    
    /**
     * @brief Initialize neighbor pointers for all spins.
     *
     * Must be called after all couplings have been defined and before
     * any Monte Carlo updates are performed.
     */
    void setup_lattice();

    /**
     * @brief Perform a Metropolis step on the specified spin
     *
     * @param T Temperature
     * @param spin the spin to be updated
     * @return 0 (rejected) or 1 (accepted)
     */
    size_t local_Metropolis(double T, HeisenbergSpin* spin);


    /**
     * @brief Perform a full Metropolis sweep over all lattice sites.
     *
     * @param T Temperature
     * @return Number of accepted updates
     */
    size_t sweep_local_Metropolis(double T);

};


// saves curent orientation of all spins to the file in `file_path`.
void save_spin_state(const Lattice& lat, const std::filesystem::path& file_path);


}
