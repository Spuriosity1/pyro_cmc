#include <algorithm>
#include <cmath>
#include <memory>

#include "MC.hpp"
#include "cli_bits.hpp"
#include "ssf_manager.hpp"
#include "energy_manager.hpp"
#include "pyrochlore_geometry.hpp"
#include "format_bits.hpp"

/*
This program performs classical Monte Carlo parallel tempering (replica
exchange) for a spin model on the pyrochlore (diamond-based) lattice.
Unlike anneal.cpp, every replica is held at a fixed temperature for the
whole run; what migrates between temperatures is the spin configuration
itself, swapped between adjacent rungs of a temperature ladder according to
the standard Metropolis exchange criterion. The simulation:

1. Constructs n_steps independent cubic supercells of the diamond lattice,
   one per replica, each pinned to its own rung of a temperature ladder
   spanning T_hot to T_cold (with --T_sample values landing exactly on
   rungs).
2. Burns each replica in independently at its own temperature.
3. Alternates sweeps-per-replica with checkerboard replica-exchange
   attempts between neighbouring rungs.
4. Samples energy at every rung, and static spin-spin correlation functions
   at the --T_sample rungs, once per exchange round.
5. Optionally saves the final spin configuration of the coldest rung.

The output consists of:

a. Static structure factor data (.ssf.h5)
b. Optionally, the final spin state of the coldest replica (.spins.h5)

 */

using namespace std;
using namespace CMC;


int main (int argc, char *argv[]) {
    ///////////////////////////////////////////////////////////////////////////
    /// Setup and CLI
    argparse::ArgumentParser prog("temper");

    /// BOOK-KEEPING
    prog.add_argument("--output_dir", "-o")
        .help("Path to output")
        .required();

    prog.add_argument("--seed", "-s")
        .required()
        .help("Seed index to seed the RNG (n_steps+1 consecutive seed indices are consumed)")
        .scan<'i', size_t>();

    prog.add_argument("--save_state")
        .help("Save the spin state of the coldest replica")
        .implicit_value(true)
        .default_value(false);

    prog.add_argument("--init_spiral")
        .help("Pre-initialise every replica's spins to a spiral with the wavevector given by --Q (requires --Q)")
        .implicit_value(true)
        .default_value(false);


    /// PARALLEL TEMPERING PROTOCOL
    prog.add_argument("--T_hot")
        .help("Temperature of the hottest replica")
        .scan<'g',double>();
    prog.add_argument("--T_cold")
        .help("Temperature of the coldest replica (defaults to smallest T_sample)")
        .scan<'g',double>();
    prog.add_argument("--T_ref")
        .help("Reference temperature for proposal distribution")
        .default_value(1.0)
        .scan<'g',double>();
    prog.add_argument("--T_sample")
        .help("Temperatures at which to accumulate the static structure factor")
        .nargs(argparse::nargs_pattern::at_least_one)
        .required()
        .scan<'g',double>();
    prog.add_argument("--n_steps")
        .help("Number of replicas (temperature ladder rungs, including the --T_sample rungs)")
        .default_value(static_cast<size_t>(100))
        .scan<'i', size_t>();
    prog.add_argument("--n_sweep", "-w")
        .help("Number of local-update sweeps per replica between exchange attempts")
        .default_value(static_cast<size_t>(16))
        .scan<'i', size_t>();
    prog.add_argument("--n_burn_in")
        .help("Number of sweeps to run at each replica's own T before exchange attempts begin")
        .default_value(static_cast<size_t>(16))
        .scan<'i', size_t>();
    prog.add_argument("--n_sample")
        .default_value(static_cast<size_t>(64))
        .help("Number of exchange rounds while collecting statistics")
        .scan<'i', size_t>();
    prog.add_argument("--prefix")
        .default_value("run");

    declare_LJ123(prog);

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
        cerr << err.what() << endl;
        cerr << prog;
        std::exit(1);
    }


    ///////////////////////////////////////////////////////////////////////////
    /// Input loading and validation

    /// Ensuring directories exist AHEAD of time (avoids heartbreak)
    std::string outdir_s = prog.get<std::string>("output_dir");
    filesystem::path outdir(outdir_s);
    if (! filesystem::exists(outdir) ){
        throw runtime_error("Cannot open outdir");
    }

    size_t seed = prog.get<size_t>("--seed");

    const double T_hot = prog.is_used("--T_hot") ?
        prog.get<double>("--T_hot") : 10;
    const std::vector<double> T_sample = prog.get<std::vector<double>>("--T_sample");
    double T_cold = prog.is_used("--T_cold") ?
        prog.get<double>("--T_cold") : *std::min_element(T_sample.begin(), T_sample.end());

    const size_t n_steps = prog.get<size_t>("--n_steps");
    const size_t n_sweep = prog.get<size_t>("--n_sweep");
    const size_t n_burn_in = prog.get<size_t>("--n_burn_in");
    const size_t n_sample = prog.get<size_t>("--n_sample");

    if (n_steps <= T_sample.size())
        throw runtime_error("--n_steps must exceed the number of --T_sample values");

    // generate_T_profile inserts T_sample exactly into the ladder, so ask it
    // for (n_steps - T_sample.size()) interpolated rungs to land on exactly
    // n_steps replicas overall.
    auto [T_grid, S_idx] = generate_T_profile(T_hot, T_cold, T_sample, n_steps - T_sample.size());
    const size_t n_replicas = T_grid.size();

    if (n_replicas < 2)
        throw runtime_error("Parallel tempering requires at least 2 replicas");

    ///////////////////////////////////////////////////////////////////////////
    /// Replica construction. Each replica owns an independent Lattice; the
    /// MC_runner/ssf_manager bound to a temperature rung are repointed
    /// (O(1)) at whichever Lattice currently holds that rung's
    /// configuration, rather than copying spins between rungs.
    std::vector<std::unique_ptr<Lattice>> lattices;
    lattices.reserve(n_replicas);
    std::vector<MC_runner> mcs;
    mcs.reserve(n_replicas);

    for (size_t k=0; k<n_replicas; k++){
        lattices.push_back(std::make_unique<Lattice>(build_pyro_lat(prog)));
        mcs.push_back(build_J1J2J3_h(prog, *lattices[k], seed + k));
    }

    if (prog.get<bool>("--init_spiral")) {
        if (!prog.is_used("--Q"))
            throw runtime_error("--init_spiral requires --Q");
        double Q_rounded = round_Qz_to_supercell(prog.get<double>("--Q"), prog.get<int>("L"));
        int spiral_axis = prog.get<int>("--spiral_axis");
        printf("Pre-initialising %zu replicas to spiral order (Q=%.10g, axis=%d)...\n",
                n_replicas, Q_rounded, spiral_axis);
        for (auto& lat_ptr : lattices)
            init_spiral_state(*lat_ptr, Q_rounded, spiral_axis);
    }

    auto B = prog.get<std::vector<double>>("-B");
    if (B.size() < 3){
        throw std::runtime_error("--B requires 3 arguments");
    }
    // Parameter specification complete. Set the name...
    std::stringstream name; // accumulates hashed options
    name << prog.get<std::string>("--prefix")<<DELIM<<name_LJ123(prog)<<
        "B="<<B[0]<<","<<B[1]<<","<<B[2]<<DELIM<<
        "seed="<<seed<<DELIM<<
        "Tc="<<T_cold<<DELIM<<
        "sw="<<n_sweep<<DELIM;

    printf("Burning in %zu replicas (%zu sweeps each)...\n", n_replicas, n_burn_in);
    for (size_t k=0; k<n_replicas; k++){
        for (size_t i=0; i<n_burn_in; i++){
            mcs[k].sweep_local_Metropolis(T_grid[k]);
        }
    }

    energy_manager e_manager;
    for (size_t k=0; k<n_replicas; k++) e_manager.new_T(T_grid[k]);

    // One shared ssf_manager: it pays FFTW_MEASURE planning cost once, then
    // for each sampled rung we point_at() the Lattice currently holding that
    // rung's configuration before sampling. This matches the existing
    // single-manager "/ssf" output layout exactly.
    ssf_manager ssfm(*lattices[0], {"xx", "yy", "zz"}, S_idx.size(), true);
    for (size_t idx : S_idx) ssfm.new_T(T_grid[idx]);

    printf("Done. Begin parallel tempering (%zu rungs, %zu of which are sampled)...\n",
            n_replicas, S_idx.size());

    XoshiroCpp::Xoroshiro128PlusPlus swap_rng(hash_true_random(seed + n_replicas));
    std::uniform_real_distribution<double> rand01(0,1);

    size_t total_swap_attempts = 0, total_swap_accepts = 0;
    std::vector<size_t> pair_attempts(n_replicas-1, 0), pair_accepts(n_replicas-1, 0);

    for (size_t round=0; round<n_sample; round++){
        for (size_t k=0; k<n_replicas; k++){
            for (size_t n=0; n<n_sweep; n++){
                mcs[k].sweep_local_Metropolis(T_grid[k]);
            }
        }

        // Checkerboard exchange: alternate which neighbouring pairs attempt
        // a swap each round, so every adjacent pair gets attempted.
        size_t start = round % 2;
        for (size_t a=start; a+1<n_replicas; a+=2){
            total_swap_attempts++;
            pair_attempts[a]++;
            const double Np = static_cast<double>(lattices[a]->lattice.num_primitive_cells());
            const double Ea = mcs[a].total_energy_per_unit_cell() * Np;
            const double Eb = mcs[a+1].total_energy_per_unit_cell() * Np;
            const double dbeta = 1.0/T_grid[a] - 1.0/T_grid[a+1];
            const double arg = dbeta * (Ea - Eb);
            if (arg >= 0 || rand01(swap_rng) < std::exp(arg)){
                std::swap(lattices[a], lattices[a+1]);
                mcs[a].rebind(*lattices[a]);
                mcs[a+1].rebind(*lattices[a+1]);
                total_swap_accepts++;
                pair_accepts[a]++;
            }
        }

        for (size_t k=0; k<n_replicas; k++){
            e_manager.set_T(T_grid[k]);
            e_manager.sample(mcs[k].total_energy_per_unit_cell());
        }
        for (size_t idx : S_idx){
            ssfm.point_at(*lattices[idx]);
            ssfm.set_T(T_grid[idx]);
            ssfm.sample();
        }

        printf("[temper] round %4zu/%zu  swap_accept=%.1f%%  E(T=%.3e)=%.6f\n",
                round+1, n_sample,
                100.0*total_swap_accepts/std::max<size_t>(1,total_swap_attempts),
                T_grid.back(), mcs[n_replicas-1].total_energy_per_unit_cell());
    }

    printf("\n[temper] per-pair swap acceptance (rung a <-> a+1):\n");
    for (size_t a=0; a+1<n_replicas; a++){
        printf("  a=%4zu  T=%.4e <-> T=%.4e  accept=%5.1f%%  (%zu/%zu)\n",
                a, T_grid[a], T_grid[a+1],
                100.0*pair_accepts[a]/std::max<size_t>(1,pair_attempts[a]),
                pair_accepts[a], pair_attempts[a]);
    }

    auto file_path = outdir/( name.str() + ".out.h5");

    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }

    ssfm.write_group(file_id, "/ssf");
    e_manager.write_group(file_id, "/energy");
    write_geometry_group(file_id, *lattices[0]);
    std::cout<<"Saved to \n"<< file_path<<std::endl;

    if (prog.get<bool>("--save_state")){
        size_t idx_cold = std::min_element(T_grid.begin(), T_grid.end()) - T_grid.begin();
        auto f = outdir /( name.str() + ".spins.h5" );
        printf("Saving spin state of coldest replica (T=%.3e) to %s\n", T_grid[idx_cold], f.string().c_str());
        save_spin_state(*lattices[idx_cold], f);
    }

    return 0;
}
