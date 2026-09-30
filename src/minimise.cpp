#include <algorithm>

#include "MC.hpp"
#include "cli_bits.hpp"
#include "ssf_manager.hpp"
#include "energy_manager.hpp"
#include "pyrochlore_geometry.hpp"
#include "format_bits.hpp"
#include "h5_bits.hpp"


/*
This program performs gradient descent for a spin
model on the pyrochlore (diamond-based) lattice, using local Metropolis updates.
The simulation:

1. Constructs a cubic supercell of the diamond lattice.
2. Defines exchange couplings up to third neighbors.
3. Runs stochastic minimisation (re-alignment of random spins)
6. Optionally saves the final spin configuration.

The output consists of:

a. Static structure factor data (.ssf.h5)
b. Optionally, the final spin state (.spins.h5)

 */

using namespace std;
using namespace CMC;

void provide_bookkeeping_args(argparse::ArgumentParser& prog){

    /// BOOK-KEEPING
    prog.add_argument("--output_dir", "-o")
        .help("Path to output")
        .required();

    prog.add_argument("--seed", "-s")
        .required()
        .help("Seed index to seed the RNG")
        .scan<'i', size_t>();

    prog.add_argument("--save_state")
        .implicit_value(true)
        .default_value(false);

    prog.add_argument("--save_state_ft")
        .implicit_value(true)
        .default_value(false);

    prog.add_argument("--init_spiral")
        .help("Pre-initialise spins to a spiral with the wavevector given by --Q (requires --Q)")
        .implicit_value(true)
        .default_value(false);

    prog.add_argument("--prefix")
        .default_value("min");
}



int main (int argc, char *argv[]) {

    ///////////////////////////////////////////////////////////////////////////
    /// Setup and CLI
    argparse::ArgumentParser prog("minimise");

    provide_bookkeeping_args(prog);

    // MINIMISER CHOICE
    prog.add_argument("--algorithm", "-a")
        .help("Minimiser: 'adaptive' (adaptive-step gradient flow), "
              "'align' (best-alignment Gauss-Seidel sweeps, sequential) [default], "
              "'newton' (truncated Riemannian Newton, trust-region), "
              "'fixed' (legacy fixed-step gradient descent)")
        .default_value(std::string("align"))
        .choices("adaptive", "align", "newton", "fixed");

    prog.add_argument("--niter", "-n")
        .help("Maximum number of minimiser iterations (steps/sweeps) to run")
        .default_value((int) 100)
        .scan<'i', int>();

    prog.add_argument("--step_size")
        .help("Initial step size. For 'adaptive' this is only a starting guess "
              "(auto-tuned); for 'fixed' it is the constant step.")
        .default_value(0.01)
        .scan<'g', double>();

    prog.add_argument("--atol")
        .help("Absolute tolerance for the adaptive step controller and the "
              "gradient-norm convergence test")
        .default_value(1e-8)
        .scan<'g', double>();

    prog.add_argument("--rtol")
        .help("Relative tolerance for the adaptive step controller and the "
              "gradient-norm convergence test")
        .default_value(1e-6)
        .scan<'g', double>();

    prog.add_argument("--tr_radius")
        .help("Initial trust-region radius for the 'newton' minimiser "
              "(auto-adapted thereafter)")
        .default_value(0.2)
        .scan<'g', double>();

    prog.add_argument("--max_inner")
        .help("Cap on truncated-CG iterations per outer step ('newton' only)")
        .default_value((int) 50)
        .scan<'i', int>();

    prog.add_argument("--fuzz")
        .help("Amplitude of gaussian noise to add to the initial condition")
        .default_value(0.)
        .scan<'g', double>();

    provide_physical_args(prog);

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
        cerr << err.what() << endl;
        cerr << prog;
        std::exit(1);
    }

    auto outdir = ensure_odir_exists(prog);

    auto lat = build_pyro_lat(prog);
    auto mc = build_J1J2J3_h<Minimiser>
        (prog, lat, prog.get<size_t>("--seed"));

    mc.randomize_spins();

    if (prog.get<bool>("--init_spiral")) {
        if (!prog.is_used("--Q"))
            throw runtime_error("--init_spiral requires --Q");
        // double Q_given = round_Qz_to_supercell(prog.get<double>("--Q"), prog.get<int>("L"));
        double Q_given = prog.get<double>("--Q");
        int spiral_axis = prog.get<int>("--spiral_axis");
        printf("Pre-initialising to spiral order (Q=%.10g, axis=%d)...\n", Q_given, spiral_axis);
        init_spiral_state(lat, Q_given, spiral_axis);
    }

    mc.fuzz_spins(prog.get<double>("--fuzz"));


    auto B = prog.get<std::vector<double>>("-B");
    if (B.size() < 3){
        throw std::runtime_error("--B requires 3 arguments");
    }
    // Parameter specification complete. Set the name...
    std::stringstream name; // accumulates hashed options
    name << prog.get<std::string>("--prefix")<<DELIM<<name_LJ123(prog)<<
        "B="<<B[0]<<","<<B[1]<<","<<B[2]<<DELIM<<
        "seed="<<prog.get<size_t>("--seed")<<DELIM<<
        "niter="<<prog.get<int>("--niter")<<DELIM;


    auto algo = prog.get<std::string>("--algorithm");
    auto max_iter = prog.get<int>("--niter");
    double step_size=prog.get<double>("--step_size");

    if (algo == "align") {
        mc.minimise_align(max_iter,
                prog.get<double>("--atol"),
                prog.get<double>("--rtol"));
    } else if (algo == "adaptive") {
        mc.minimise_adaptive(max_iter, step_size, prog.get<double>("--atol"),
                prog.get<double>("--rtol"));
    } else if (algo == "newton") {
        mc.minimise_newton_tr(max_iter, prog.get<double>("--atol"),
                prog.get<double>("--rtol"), prog.get<double>("--tr_radius"),
                prog.get<int>("--max_inner"));
    } else if (algo == "fixed") {
        for (int i = 0; i < max_iter; ++i) {
            mc.gradient_descent(step_size);
            double E = mc.total_energy_per_unit_cell();
            printf("Iter %4d\tE=%.3f\n", i, E);
        }
    } 

    auto file_path = outdir/( name.str() + ".out.h5");
    H5Fclose(h5_create_trunc_nolock(file_path.string()));


    // Full symmetric spin-correlation matrix: the three diagonal components give
    // the Heisenberg trace <S.S>(q); the three off-diagonals carry the spiral-plane
    // tensor (real part) and the vector chirality (imaginary part), from which the
    // O(3)-invariant plane observables are reconstructed in postprocessing.
    ssf_manager ssfm(lat, 
            {"xx", "yy", "zz", "xy", "xz", "yz"},
                     file_path.string(), "/ssf", {0}, true);

    ssfm.set_T(0);
    ssfm.sample();
    ssfm.flush();


    // ssf metadata reopens the file itself; energy + geometry share one final
    // brief reopen.
    ssfm.write_group(-1, "/ssf");
    {
        hid_t file_id = h5_open_rdwr_nolock(file_path.string());
        write_geometry_group(file_id, lat);
        H5Fclose(file_id);
    }
    std::cout<<"Saved to \n"<< file_path<<std::endl;

    if (prog.get<bool>("--save_state_ft")){
        auto f = outdir /( name.str() + ".spindft.h5" );
        printf("Saving Fourier transformed spin state to %s\n", f.string().c_str());
        save_spin_state(lat, f);
    }
    
    if (prog.get<bool>("--save_state")){
        auto f = outdir /( name.str() + ".spins.h5" );
        printf("Saving spin state to %s\n", f.string().c_str());
        save_ft_spin_state(lat, f);
    }


    return 0;

}
