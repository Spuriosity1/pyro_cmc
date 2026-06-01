#include <algorithm>
#include "cli_bits.hpp"

#include "MC.hpp"
#include "ssf_manager.hpp"
#include "energy_manager.hpp"
#include "pyrochlore_geometry.hpp"
#include "format_bits.hpp"

/*
This program performs classical MC simulated annealing for an Ising model on a random graph.
The simulation:


*/


using namespace std;
using namespace CMC;


int main (int argc, char *argv[]) {
    ///////////////////////////////////////////////////////////////////////////
    /// Setup and CLI
    argparse::ArgumentParser prog("random_ice");

    /// BOOK-KEEPING
    prog.add_argument("--output_dir", "-o")
        .help("Path to output")
        .required();

    prog.add_argument("--seed", "-s")
        .required()
        .help("64-bit int to seed the RNG");

    prog.add_argument("--save_state")
        .implicit_value(true)
        .default_value(false);


    /// Ensuring directories exist AHEAD of time (avoids heartbreak)
    std::string outdir_s = prog.get<std::string>("output_dir");
    filesystem::path outdir(outdir_s);
    if (! filesystem::exists(outdir) ){
        throw runtime_error("Cannot open outdir");
    }


    auto supercell_spec = imat33_t::from_cols({L,0,0},{0,L,0},{0,0,L});
    auto cell_spec = PyroCubicCell();
    return build_supercell(cell_spec, supercell_spec);




    return 0;
}
