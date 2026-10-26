#pragma once
#include "format_bits.hpp"
#include "vec3.hpp"
#include <algorithm>
#include <argparse/argparse.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <ostream>
#include <type_traits>

#include "vec3.hpp"
#include "random_seeds.hpp"
#include "pyrochlore_geometry.hpp"



using mat33d=vector3::mat33<double>;

namespace coupling {
static const mat33d Heis = mat33d::from_cols(
        {1,0,0},{0,1,0},{0,0,1});

static const mat33d Isin = mat33d::from_cols(
        {0,0,0},{0,0,0},{0,0,1});
};

// Magnetoelastic tetragonal distortion along a cubic axis (0=x, 1=y, 2=z).
//
// A tetragonal distortion scales space along `axis` by a factor `s`, changing
// every bond length and hence every exchange. Following the notebook
// strain_bond_table() helper, a bond with (unstrained) integer displacement
// `disp` has its coupling scaled by
//
//     J -> J * (1 + tetra * f),     f = 1 - |disp . scale| / |disp|,
//
// with scale = (1,1,s) for axis==2 (and cyclic). `f` is the fractional reduction
// of the bond length, so `tetra` is the linear magnetoelastic response and `s`
// sets the distortion shape. This applies to *every* shell (J1..J4): bonds
// perpendicular to `axis` are unchanged (f=0) and the others are strengthened in
// proportion to their projection onto `axis`. `tetra == 0` is the isotropic
// limit. For the six nn1 (tetrahedron-edge) bonds this reproduces the original
// "strengthen 4 of 6" split, the four axial bonds sharing one factor and the two
// perpendicular ones another.
inline double tetra_bond_scale(const ipos_t& disp, int axis, double tetra, double s) {
    double d2 = 0.0, ds2 = 0.0;
    for (int i = 0; i < 3; i++) {
        const double c  = static_cast<double>(disp[i]);
        const double sc = (i == axis) ? s : 1.0;
        d2  += c * c;
        ds2 += (sc * c) * (sc * c);
    }
    if (d2 == 0.0) return 1.0;
    const double f = 1.0 - std::sqrt(ds2 / d2);
    return 1.0 + tetra * f;
}


inline auto provide_physical_args(argparse::ArgumentParser& prog){

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
        .help("Third-nearest-neighbour Heisenberg coupling strength (mutually exclusive with --Q)")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--J4")
        .help("Fourth-nearest-neighbour Heisenberg coupling strength")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--K")
        .help("Biquadratic interaction strength")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--D")
        .help("Long-range dipolar interaction strength (Ewald/Madelung summed), in the "
              "same energy units as the exchange J's. 0 disables it (zero overhead).")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--ewald_alpha")
        .help("Ewald splitting parameter alpha (1/length, cubic cell = 1 unit). "
              "Omit for an automatic choice; results are alpha-independent.")
        .default_value(-1.0)
        .scan<'g', double>();

    prog.add_argument("--Q")
        .help("Nonzero spiral wavevector component in units of 2pi/a_cubic; J3 is set to minimise the spiral "
              "energy given J2, J4 and K (mutually exclusive with --J3). Rounded to nearest supercell-commensurate value.")
        .scan<'g', double>();
    prog.add_argument("--spiral_axis", "-x")
        .help("Axis [0,1,2] (x,y,z) that the spiral wavevector --Q points along")
        .choices(0,1,2)
        .default_value(2)
        .scan<'i', int>();
    prog.add_argument("--Jzz")
        .help("Nearest-neighbour XXZ anisotropy (local [111] frame),  Jzz * Sz Sz."
              "Jzz=0 is isotropic Heisenberg.")
        .default_value(0.0)
        .scan<'g', double>();
    prog.add_argument("--tetragonal")
        .help("Magnetoelastic tetragonal distortion strength (linear response). Every "
              "coupling J1..J4 is scaled by J*(1 + tetragonal*f), where f is the "
              "fractional bond-length change under a scaling of --tetra_axis by "
              "--tetra_scale. Bonds perpendicular to the axis are unchanged. "
              "0 is the isotropic limit.")
        .default_value(0.0)
        .scan<'g', double>();
    prog.add_argument("--tetra_axis")
        .help("Axis [0,1,2] (x,y,z) of the tetragonal distortion applied by --tetragonal")
        .choices(0,1,2)
        .default_value(2)
        .scan<'i', int>();
    prog.add_argument("--tetra_scale")
        .help("Geometric scale factor applied along --tetra_axis to define the "
              "tetragonal distortion shape (bond-length change f). Default 0.98 "
              "matches the notebook strain_bond_table helper.")
        .default_value(0.98)
        .scan<'g', double>();
    prog.add_argument("--external_field", "-B")
        .help("Global magnetic field")
        .nargs(3)
        .default_value(std::vector<double>({0.,0.,0.}))
        .scan<'g', double>();
}

// Compute the J3 that makes a (0,0,Q) spiral stationary in the single-Q spiral
// energy, given J2, J4 and the nn1 biquadratic K (all in the ferromagnetic
// convention |J1|=1). Qz is in units of 2pi/a_cubic. Only J3 enters the cos(4q)
// term, so the stationarity condition stays linear in J3 and inverts in closed
// form. J4 adds a cos(6q) contribution, hence the c^2 term below.
//
// The nn1 biquadratic (K/2)(S_i.S_j)^2 evaluates to (K/2)cos^2(q d_z) on the
// spiral, and cos^2 = (1+cos(2q d_z))/2 feeds the SAME cos(4q) harmonic as J3
// (the nn1 shell has d_z in {0,+-2}). It therefore only shifts the required J3
// by a constant: J3(K) = J3(0) - K/8, independent of Q, J2, J4. Derivation and
// numerical check: scripts/spiral_Q.sage.
// At J4=K=0 this reduces to the original (-4J2 + (1-2J2)/cos2q)/8.
inline double J3_from_Qz(double J2, double J4, double K, double Qz) {
    Qz *= 2 * M_PI / 8; // this 8 is from the conventional pyrochlore cell size
    const double c = std::cos(2*Qz);
    return -(12*J4*c*c + 4*J2*c + 2*J2 - 2*J4 - 1) / (8*c) - K / 8;
}

// Round Qz (in units of 1/a_cubic) to the nearest value commensurate with an
// L-cell supercell (i.e. to a multiple of 2pi/L).
inline double round_Qz_to_supercell(double Qz, int L) {
    const double step = 1. / L;
    return std::round(Qz / step) * step;
}

// Warn if the rounding of Qz to the supercell grid exceeds min(1e-10, 1e-3*|Qz|).
inline void warn_Qz_rounding(double Qz, double Qz_rounded) {
    double diff = std::abs(Qz - Qz_rounded);
    double tol  = std::min(1e-10, 1e-3 * std::abs(Qz));
    if (diff > tol) {
        fprintf(stderr,
            "WARNING: Qz=%.10g rounded to %.10g (diff=%.3e > tol=%.3e)\n",
            Qz, Qz_rounded, diff, tol);
    }
}

// Extract the effective J3 from the parsed arguments, handling --Q or --J3.
inline double resolve_J3(const argparse::ArgumentParser& prog) {
    bool has_J3 = prog.is_used("--J3");
    bool has_Q = prog.is_used("--Q");
    if (has_J3 && has_Q)
        throw std::runtime_error("--J3 and --Q are mutually exclusive");
    if (has_Q) {
        double Qz =prog.get<double>("--Q");
        double J1 = prog.get<double>("--J1");
        double J2_eff = prog.get<double>("--J2") / std::abs(J1);
        double J4_eff = prog.get<double>("--J4") / std::abs(J1);
        double K_eff  = prog.get<double>("--K")  / std::abs(J1);
        return J3_from_Qz(J2_eff, J4_eff, K_eff, Qz) * std::abs(J1);
    }
    return prog.get<double>("--J3");
}


inline uint64_t int_from_hex_str(const std::string& seed_s){
    uint64_t seed; // ugly hack for loading seed as hex
    std::stringstream ss;
    ss << std::hex << seed_s;
    ss >> seed; 
    return seed;
}

inline auto build_pyro_lat(const argparse::ArgumentParser& prog){
    int L = prog.get<int>("L");
    auto supercell_spec = imat33_t::from_cols({L,0,0},{0,L,0},{0,0,L});
    auto cell_spec = PyroCubicCell();
    return build_supercell(cell_spec, supercell_spec);
}

inline uint64_t hash_true_random(uint64_t seed_id){
    if (seed_id > true_random_cache.size()){
        throw std::runtime_error("Cannot read: there are only "+std::to_string(true_random_cache.size())+"seeds");
    }
    return true_random_cache[seed_id];
}

template<typename T>
requires std::is_base_of_v<CMC::InteractingHamiltonian, T>
auto build_J1J2J3_h(const argparse::ArgumentParser& prog, CMC::Lattice& lat, uint64_t seed){

    vector3::vec3d global_field;
    {
        auto B_tmp = prog.get<std::vector<double>>("-B");
        for (int i=0; i<3; i++){ global_field[i]= B_tmp[i]; }
    }

    auto J1 = prog.get<double>("--J1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = resolve_J3(prog);
    if (prog.is_used("--Q")) {
        double Qz         = prog.get<double>("--Q");
        int axis = prog.get<int>("--spiral_axis");
        printf("Using Q=%.10g along axis %d -> J3=%.10g\n", Qz, axis, J3);
    }

    auto Jzz = prog.get<double>("--Jzz");
    auto tetra       = prog.get<double>("--tetragonal");
    auto tetra_axis  = prog.get<int>("--tetra_axis");
    auto tetra_scale = prog.get<double>("--tetra_scale");

    T mc(lat, hash_true_random(seed));

    // J1 as six sublattice-pair specs for local-frame XXZ: exactly the six
    // nn1 bonds of one tetrahedron, declared once each (de-duplicated).
    // For pair (mu, nu) with mu < nu, the source-applied matrix is
    //   J_spec = J1 * I + Jzz * z_mu ⊗ z_nu.
    // so the lower-sublattice (source) field is h_mu += J_spec * S_nu, giving the
    // bond energy J1 S_mu·S_nu + Jzz (S_mu·z_mu)(S_nu·z_nu) . The target
    // (nu) endpoint references J_spec^T (stored as CouplingSpec::Jt).
    {
        using vec3d = vector3::vec3<double>;
        static const std::pair<int,int> pairs[6] =
            {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
        static const std::vector<std::vector<ipos_t>>* nn1_pairs[6] = {
            &pyrochlore::nn1_pair_01, &pyrochlore::nn1_pair_02, &pyrochlore::nn1_pair_03,
            &pyrochlore::nn1_pair_12, &pyrochlore::nn1_pair_13, &pyrochlore::nn1_pair_23
        };
        for (int k = 0; k < 6; k++) {
            auto [mu, nu] = pairs[k];
            const vec3d& z_mu = pyrochlore::axis[mu][2];
            const vec3d& z_nu = pyrochlore::axis[nu][2];
            // (z_mu ⊗ z_nu)_{ab} = z_mu[a]*z_nu[b]; column b is z_nu[b]*z_mu.
            // mat33d::from_cols takes std::array<double,3>, so convert explicitly
            auto v2a = [](vec3d v) -> std::array<double,3> {
                return {v[0], v[1], v[2]};
            };
            mat33d local_Ising = mat33d::from_cols(
                v2a(z_nu[0] * z_mu), v2a(z_nu[1] * z_mu), v2a(z_nu[2] * z_mu));
            std::string pname = "J1_" + std::to_string(mu) + std::to_string(nu);
            // Tetragonal distortion scales this edge's J1 by its bond-length change.
            // Both entries of the pair table are +-v, so the first displacement is
            // a faithful representative of the bond direction.
            ipos_t disp{};
            for (const auto& sub : *nn1_pairs[k]) { if (!sub.empty()) { disp = sub.front(); break; } }
            double J1_bond = J1 * tetra_bond_scale(disp, tetra_axis, tetra, tetra_scale);
            mc.define_general_coupling(pname, *nn1_pairs[k], Jzz * local_Ising + J1_bond*coupling::Heis);
        }
    }
    // Further-neighbour Heisenberg shells are keyed by squared spin separation
    // (cubic cell side = 8): J2 -> 24, J3 -> 32, J4 -> 40. The old nn3a/nn3b
    // split folds into the single dist_sq=32 shell (both shared J3).
    //
    // Under a tetragonal distortion each shell is split into sub-groups by the
    // bond-length change f (bonds sharing a factor are registered together), and
    // every sub-group is scaled by J*(1+tetra*f) -- the same magnetoelastic rule
    // the notebook strain_bond_table() applies to J1..J4.
    auto define_tetra_Heisenberg = [&](const std::string& name,
            const std::vector<std::vector<ipos_t>>& shell, double J) {
        if (tetra == 0.0) { mc.define_Heisenberg_coupling(name, shell, J); return; }
        std::map<long long, std::vector<std::vector<ipos_t>>> by_factor;
        for (std::size_t sl = 0; sl < shell.size(); sl++) {
            for (const auto& d : shell[sl]) {
                double fac = tetra_bond_scale(d, tetra_axis, tetra, tetra_scale);
                auto& tbl = by_factor[std::llround(fac * 1e9)];
                if (tbl.empty()) tbl.resize(shell.size());
                tbl[sl].push_back(d);
            }
        }
        int g = 0;
        for (auto& [key, tbl] : by_factor)
            mc.define_Heisenberg_coupling(name + "_t" + std::to_string(g++),
                                          tbl, J * (static_cast<double>(key) * 1e-9));
    };
    auto J4 = prog.get<double>("--J4");
    define_tetra_Heisenberg("J2", pyrochlore::neighbours_by_distance(24), J2);
    define_tetra_Heisenberg("J3", pyrochlore::neighbours_by_distance(32), J3);
    if (J4 != 0)
        define_tetra_Heisenberg("J4", pyrochlore::neighbours_by_distance(40), J4);

    // Nearest-neighbour biquadratic K/2 (S_i·S_j)^2. Only registered when
    // nonzero so pure-linear runs keep biquad_bonds empty (zero overhead).
    auto K = prog.get<double>("--K");
    if (K != 0)
        mc.define_biquad_coupling("K", pyrochlore::nn1_dist, K);

    mc.set_global_field(global_field);

    mc.setup_lattice();

    // Long-range dipolar interactions. The Ewald tensor is built here (once);
    // the field cache is filled from the current spins and is refreshed again by
    // randomize_spins()/init, so the order of this call vs. the initial state is
    // immaterial. Disabled (and zero-cost) when --D is 0.
    auto D = prog.get<double>("--D");
    if (D != 0) {
        CMC::EwaldParams ep;
        double ealpha = prog.get<double>("--ewald_alpha");
        if (ealpha > 0) ep.alpha = ealpha;
        mc.enable_dipolar(D, ep);
    }

    return mc;
}

// Initialise all spins to a spiral with wavevector Q along the given axis
// (0=x, 1=y, 2=z; Q_rounded in units of 2pi/a_cubic). Spins rotate in the plane normal to Q.
inline void init_spiral_state(CMC::Lattice& lat, double Q_rounded, int axis) {
    constexpr double a_cubic = 8.;
    int idx_sin = (axis + 1) % 3;
    int idx_cos = (axis + 2) % 3;
    for (auto& s : lat.get_objects<CMC::HeisenbergSpin>()) {
        double phase = Q_rounded * 2 * M_PI * static_cast<double>(s.ipos[axis]) / a_cubic;
        vector3::vec3d S(0., 0., 0.);
        S[idx_cos] = std::cos(phase);
        S[idx_sin] = std::sin(phase);
        s.S = S;
    }
}

inline auto name_LJ123(const argparse::ArgumentParser& prog){
    auto J1 = prog.get<double>("--J1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = resolve_J3(prog);
    auto J4 = prog.get<double>("--J4");
    auto Jzz = prog.get<double>("--Jzz");
    auto K = prog.get<double>("--K");
    auto tetra = prog.get<double>("--tetragonal");
    int L = prog.get<int>("L");

    std::stringstream name;
    name << "L="<<L<<DELIM<<
        "J1="<<J1<<DELIM<<
        "J2="<<J2<<DELIM<<
        "J3="<<J3<<DELIM;
    if (J4 != 0)
        name << "J4="<<J4<<DELIM;
    name <<
        "Jzz="<<Jzz<<DELIM<<
        "K="<<K<<DELIM;
    if (tetra != 0) {
        name << "tetra="<<tetra<<DELIM;
        int tetra_axis = prog.get<int>("--tetra_axis");
        if (tetra_axis != 2)
            name << "tetra_axis="<<tetra_axis<<DELIM;
        double tetra_scale = prog.get<double>("--tetra_scale");
        if (tetra_scale != 0.98)
            name << "tetra_scale="<<tetra_scale<<DELIM;
    }
    if (prog.is_used("--Q")) {
        double Qz = prog.get<double>("--Q");
        name << "Q="<<Qz<<DELIM;
        int axis = prog.get<int>("--spiral_axis");
        if (axis != 2)
            name << "spiral_axis="<<axis<<DELIM;
    }
    return name.str();
}

// Generates a log-spaced array from T_hot to min(T_sample), making sure to
// include all of the T_sample steps exactly in sorted order.
// Returns also the 
inline std::pair<std::vector<double>, std::set<size_t>> generate_T_profile(
        double T_hot, double T_cold,
        std::vector<double> T_sample, size_t total_steps){
    std::vector<double> T_sweep, T;
    const double factor = pow(T_cold / T_hot, 1./total_steps);
    double t = T_hot/factor;
    for (size_t i=0; i<total_steps; i++){
        t *= factor;
        T_sweep.push_back(t);
    }

    // sort both descending
    std::sort(T_sweep.begin(),T_sweep.end(), [](double a, double b){return a>b;}); 
    std::sort(T_sample.begin(),T_sample.end(), [](double a, double b){return a>b;}); 
    
    size_t i_samp=0;
    size_t i_sw=0;

    std::set<size_t> idx;
    while(i_sw < T_sweep.size() && i_samp < T_sample.size()){
        if (T_sweep[i_sw] > T_sample[i_samp]){
            T.push_back(T_sweep[i_sw]); i_sw++;
        } else {
            idx.insert(T.size());
            T.push_back(T_sample[i_samp]); i_samp++;
        }
    }
    
    // exhaust any tails
    for (; i_samp < T_sample.size(); i_samp++){
        idx.insert(T.size());
        T.push_back(T_sample[i_samp]);
    }
    for (; i_sw < T_sweep.size(); i_sw++)
        T.push_back(T_sweep[i_sw]);


    return {T, idx};

}

inline auto ensure_odir_exists(argparse::ArgumentParser& prog){
    /// Ensuring directories exist AHEAD of time (avoids heartbreak)
    std::string outdir_s = prog.get<std::string>("output_dir");
    std::filesystem::path outdir(outdir_s);
    if (! std::filesystem::exists(outdir) ){
        throw std::runtime_error("Cannot open outdir");
    }
    return outdir;
}



