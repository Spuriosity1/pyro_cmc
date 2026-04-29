#include <algorithm>
#include <argparse.hpp>
#include <filesystem>
#include <iostream>
#include <ostream>


#include "MC.hpp"
#include "stats.hpp"
#include "energy_manager.hpp"
#include "pyrochlore_geometry.hpp"
#include "format_bits.hpp"

/*
This program performs classical Monte Carlo simulated annealing for a spin
model on the pyrochlore (diamond-based) lattice, using local Metropolis updates.
The simulation:

1. Constructs a cubic supercell of the diamond lattice.
2. Defines exchange couplings up to third neighbors.
3. Performs a burn-in at high temperature.
4. Anneals the system from a hot temperature to a target cold temperature.
5. Samples static spin–spin correlation functions at the final temperature.
6. Optionally saves the final spin configuration.

The output consists of:

a. Static structure factor data (.ssf.h5)
b. Optionally, the final spin state (.spins.h5)

 */

using namespace std;
using namespace CMC;

using mat33d=vector3::mat33<double>;

namespace coupling {
static const mat33d Heis = mat33d::from_cols(
        {1,0,0},{0,1,0},{0,0,1});

static const mat33d Isin = mat33d::from_cols(
        {0,0,0},{0,0,0},{0,0,1});
};

int main (int argc, char *argv[]) {
    ///////////////////////////////////////////////////////////////////////////
    /// Setup and CLI
    argparse::ArgumentParser prog("anneal");

    /// BOOK-KEEPING
    prog.add_argument("--output_dir", "-o")
        .help("Path to output")
        .required();

    std::string seed_s;
    prog.add_argument("--seed", "-s")
        .required()
        .help("64-bit int to seed the RNG")
        .store_into(seed_s);

    prog.add_argument("--save_state")
        .implicit_value(true)
        .default_value(false);

    /// ANNEALING PROTOCOL
    prog.add_argument("--T_hot")
        .help("Temperature  to begin annealing from")
        .scan<'g',double>();
    prog.add_argument("--T_ref")
        .help("Reference temperature for proposal distribution")
        .default_value(1.0)
        .scan<'g',double>();
    prog.add_argument("--T_cold")
        .help("Final temperature")
        .required()
        .scan<'g',double>();
    prog.add_argument("--n_steps")
        .help("Number of annealing steps")
        .default_value(static_cast<size_t>(100))
        .scan<'i', size_t>();
    prog.add_argument("--n_sweep")
        .help("Number of sweeps to run per temperatur step")
        .default_value(static_cast<size_t>(16))
        .scan<'i', size_t>();
    prog.add_argument("--n_burn_in")
        .help("Number of sweeps to run at T_hot before we start annealing")
        .default_value(static_cast<size_t>(16))
        .scan<'i', size_t>();
    prog.add_argument("--n_sample")
        .default_value(static_cast<size_t>(64))
        .help("Number of sweeps to run at T_cold while collecting statistics")
        .scan<'i', size_t>();

    /// PHYSICAL
    prog.add_argument("L")
        .help("Linear dimension of cubic supercell (e.g. L=2 has 2^3 *4 = 32 primitive cells)")
        .scan<'i', int>(); // cubic unit cell hardcoded
    prog.add_argument("--J1")
        .help("Nearest-neighbour Heisenberg coupling strength")
        .required()
        .scan<'g', double>();
    prog.add_argument("--J2")
        .help("Second-nearest-neighbour Heisenberg coupling strength")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--J3")
        .help("Third-nearest-neighbour Heisenberg coupling strength")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--external_field", "-B")
        .help("Global magnetic field")
        .nargs(3)
        .default_value(std::vector<double>({0.,0.,0.}))
        .scan<'g', double>();


    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
        cerr << err.what() << endl;
        cerr << prog;
        std::exit(1);
    }


    ///////////////////////////////////////////////////////////////////////////
    /// Input loading and validation
    
    std::stringstream name; // accumulates hashed options

    /// Ensuring directories exist AHEAD of time (avoids heartbreak)
    std::string outdir_s = prog.get<std::string>("output_dir");
    filesystem::path outdir(outdir_s);
    if (! filesystem::exists(outdir) ){
        throw runtime_error("Cannot open outdir");
    }


    uint64_t seed; // ugly hack for loading seed as hex
    std::stringstream ss;
    ss << std::hex << seed_s;
    ss >> seed; 


    int L = prog.get<int>("L");
    auto supercell_spec = imat33_t::from_cols({-L, L, L}, {L, -L, L}, {L, L, -L});
    auto cell_spec = DiamondSpinSpec();
    CMC::Lattice lat = build_supercell(cell_spec, supercell_spec);

    auto J1 = prog.get<double>("--J1");
//    auto K1 = prog.get<double>("--K1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = prog.get<double>("--J3");
    vector3::vec3d global_field;
    { 
        auto B_tmp = prog.get<std::vector<double>>("-B");
        for (int i=0; i<3; i++){ global_field[i]= B_tmp[i]; }
        cout<<"B="<<global_field<<std::endl;
    }


    CMC::MC_runner mc(lat, seed);
    mc.define_coupling("J1", pyrochlore::nn1_dist, 
        mat33d::from_cols({J1, 0,0}, {0,J1, 0}, {0,0,J1})
        );
    mc.define_coupling("J2", pyrochlore::nn2_dist, J2*coupling::Heis);
    mc.define_coupling("J3a", pyrochlore::nn3a_dist, J3*coupling::Heis);
    mc.define_coupling("J3b", pyrochlore::nn3b_dist, J3*coupling::Heis);
    mc.define_global_field(global_field);

    mc.settings.T_ref = prog.get<double>("--T_ref");
    mc.setup_lattice();

    const double T_hot = prog.is_used("--T_hot") ? 
        prog.get<double>("--T_hot") :
        sqrt(J1*J1 + J2* J2 + J3*J3)*10;
    const double T_cold = prog.get<double>("--T_cold");

    const size_t n_steps = prog.get<size_t>("--n_steps");
    const size_t n_sweep = prog.get<size_t>("--n_sweep");
    const size_t n_burn_in = prog.get<size_t>("--n_burn_in");
    const size_t n_sample = prog.get<size_t>("--n_sample");

    double T = T_hot;

    // Parameter specification complete. Set the name...

    name <<"L="<<L<<DELIM<<
        "J1="<<J1<<DELIM<<
        "J2="<<J2<<DELIM<<
        "J3="<<J3<<DELIM<<
        "seed="<<seed_s<<DELIM<<
        "T_c="<<T_cold<<DELIM;


    printf("Burning in (%zu sweeps)...\n", n_burn_in);
    for (size_t i=0; i<n_burn_in; i++){
        mc.sweep_local_Metropolis(T_hot);
    }

    printf("Done. Begin anneal...\n");
    const double factor = pow(T_cold / T_hot, 1./n_steps);

    energy_manager e_manager;

    for (size_t i=0; i <n_steps; i++){
        size_t accepted=0;
        e_manager.new_T(T);
        for (size_t n=0; n<n_sweep; n++){
            accepted += mc.sweep_local_Metropolis(T);
        }
        double e = mc.total_energy_per_unit_cell();
        e_manager.sample(e);

        double E = e_manager.curr_E();
        printf("Iter %4zu T=%.3e E=%3e Acceptance rate: %.2f%%\n", 
                i, T, E, accepted*100.0/lat.get_objects<HeisenbergSpin>().size()/n_sweep);
        T *= factor;
    }

    static_corr_3D ssf_manager(lat);

    ssf_manager.declare_observable("SdotS", 
            static_corr_3D::NEEDS_XX | static_corr_3D::NEEDS_YY | static_corr_3D::NEEDS_ZZ );
    ssf_manager.declare_observable("SzSz", static_corr_3D::NEEDS_ZZ);

    printf("Sampling at T=%lf (%zu sweeps)...\n", T, n_sample);
    for (size_t i=0; i<n_sample; i++){
        for (size_t n=0; n<n_sweep; n++){
            mc.sweep_local_Metropolis(T);
        }
        ssf_manager.sample();
    }

    auto file_path = outdir/( name.str() + ".out.h5");

    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }
    ssf_manager.write_group(file_id, "/ssf");
    e_manager.write_group(file_id, "/energy");
    write_geometry_group(file_id, lat);
    
    if (prog.get<bool>("--save_state")){
        auto f = outdir /( name.str() + ".spins.h5" );
        printf("Saving spin state to %s\n", f.string().c_str());
        save_spin_state(lat, f);
    }



    return 0;
}
