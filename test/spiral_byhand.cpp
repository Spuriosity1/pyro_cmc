
#include <algorithm>
#include <argparse.hpp>
#include <filesystem>
#include <iostream>
#include <ostream>


#include "stats.hpp"
#include "energy_manager.hpp"
#include "pyrochlore_geometry.hpp"
#include "format_bits.hpp"

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
    argparse::ArgumentParser prog("anneal");

    prog.add_argument("--output_dir", "-o")
        .help("Path to output")
        .required();

    std::string seed_s;
    prog.add_argument("--seed", "-s")
        .required()
        .help("64-bit int to seed the RNG")
        .store_into(seed_s);

    prog.add_argument("L")
        .help("Linear dimension of cubic supercell")
        .scan<'i', int>();

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
        cerr << err.what() << endl;
        cerr << prog;
        std::exit(1);
    }

    std::stringstream name;

    std::string outdir_s = prog.get<std::string>("output_dir");
    filesystem::path outdir(outdir_s);
    if (!filesystem::exists(outdir))
        throw runtime_error("Cannot open outdir");

    uint64_t seed;
    std::stringstream ss;
    ss << std::hex << seed_s;
    ss >> seed;

    int L = prog.get<int>("L");
    double q = static_cast<double>(L);
    auto supercell_spec = imat33_t::from_cols({-L, L, L}, {L, -L, L}, {L, L, -L});

    auto cell_spec = DiamondSpinSpec();
    CMC::Lattice lat = build_supercell(cell_spec, supercell_spec);

    CMC::MC_runner mc(lat, seed);
    mc.setup_lattice();

    name << "L=" << L << DELIM
         << "q=" << q << DELIM;

    static_corr_3D ssf_manager(lat);

    ssf_manager.declare_observable("SdotS",
            static_corr_3D::NEEDS_XX | static_corr_3D::NEEDS_YY | static_corr_3D::NEEDS_ZZ);
    ssf_manager.declare_observable("SzSz", static_corr_3D::NEEDS_ZZ);

    ssf_manager.sample();

    auto file_path = outdir / (name.str() + ".out.h5");

    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC,
            H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    ssf_manager.write_group(file_id, "/ssf");

    auto f = outdir / (name.str() + ".spins.h5");
    printf("Saving spin state to %s\n", f.string().c_str());
    save_spin_state(lat, f);

    return 0;
}
