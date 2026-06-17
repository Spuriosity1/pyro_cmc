#pragma once
#include "format_bits.hpp"
#include "vec3.hpp"
#include <algorithm>
#include <argparse.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <ostream>

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


inline auto declare_LJ123(argparse::ArgumentParser& prog){

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
        .help("Third-nearest-neighbour Heisenberg coupling strength (mutually exclusive with --Qz)")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--Qz")
        .help("Spiral wavevector z-component in units of 2pi/a_cubic; J3 is set to minimise spiral energy "
              "(mutually exclusive with --J3). Rounded to nearest supercell-commensurate value.")
        .scan<'g', double>();
    prog.add_argument("--Delta")
        .help("Nearest-neighbour XXZ anisotropy (local [111] frame): Jz = Delta*J1, Jperp = J1. "
              "Delta=1 is isotropic Heisenberg.")
        .default_value(1.0)
        .scan<'g', double>();
    prog.add_argument("--external_field", "-B")
        .help("Global magnetic field")
        .nargs(3)
        .default_value(std::vector<double>({0.,0.,0.}))
        .scan<'g', double>();
}

// Compute J3 from J2 and Qz that minimises the spiral energy (assumes |J1|=1).
// Qz is in units of 2pi/a_cubic.
inline double J3_from_Qz(double J2, double Qz) {
    Qz *= 2 * M_PI / 8; // this 8 is from the conventional pyrochlore cell size

    return (-4*J2 + 1./std::cos(2*Qz) - 2.*J2/std::cos(2*Qz))/8.;
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

// Extract the effective J3 from the parsed arguments, handling --Qz or --J3.
inline double resolve_J3(const argparse::ArgumentParser& prog) {
    bool has_J3 = prog.is_used("--J3");
    bool has_Qz = prog.is_used("--Qz");
    if (has_J3 && has_Qz)
        throw std::runtime_error("--J3 and --Qz are mutually exclusive");
    if (has_Qz) {
        int    L  = prog.get<int>("L");
        double Qz = round_Qz_to_supercell(prog.get<double>("--Qz"), L);
        double J1 = prog.get<double>("--J1");
        double J2_eff = prog.get<double>("--J2") / std::abs(J1);
        double J3 = J3_from_Qz(J2_eff, Qz) * std::abs(J1);
        std::cout<<"Calculated J3="<<J3<<std::endl;
        return J3;
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


inline auto build_J1J2J3_h(const argparse::ArgumentParser& prog, CMC::Lattice& lat, uint64_t seed){

    vector3::vec3d global_field;
    {
        auto B_tmp = prog.get<std::vector<double>>("-B");
        for (int i=0; i<3; i++){ global_field[i]= B_tmp[i]; }
    }

    auto J1 = prog.get<double>("--J1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = resolve_J3(prog);
    if (prog.is_used("--Qz")) {
        int    L          = prog.get<int>("L");
        double Qz         = prog.get<double>("--Qz");
        double Qz_rounded = round_Qz_to_supercell(Qz, L);
        warn_Qz_rounding(Qz, Qz_rounded);
        printf("Using Qz=%.10g -> J3=%.10g\n", Qz_rounded, J3);
    }

    auto Delta = prog.get<double>("--Delta");

    CMC::MC_runner mc(lat, hash_true_random(seed));

    // J1 as six sublattice-pair specs for local-frame XXZ.
    // For pair (mu, nu) with mu < nu, J_spec = J1 * [I + (Delta-1) * z_nu ⊗ z_mu].
    // bonds_above/bonds_below are split by pyro_sl order so J_spec is applied
    // consistently: h_nu += J_spec * S_mu,  h_mu += J_spec^T * S_nu.
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
            // J_spec = J1*I + J1*(Delta-1)*(z_nu ⊗ z_mu)
            // column j of (z_nu ⊗ z_mu) is z_mu[j]*z_nu
            // mat33d::from_cols takes std::array<double,3>, so convert explicitly
            auto v2a = [](vec3d v) -> std::array<double,3> {
                return {v[0], v[1], v[2]};
            };
            mat33d J_spec = mat33d::from_cols(
                v2a(J1*(Delta-1)*z_mu[0]*z_nu + vec3d(J1, 0.0, 0.0)),
                v2a(J1*(Delta-1)*z_mu[1]*z_nu + vec3d(0.0, J1, 0.0)),
                v2a(J1*(Delta-1)*z_mu[2]*z_nu + vec3d(0.0, 0.0, J1)));
            std::string pname = "J1_" + std::to_string(mu) + std::to_string(nu);
            mc.define_coupling(pname, *nn1_pairs[k], J_spec, /*use_pyro_sl_ordering=*/true);
        }
    }
    mc.define_coupling("J2", pyrochlore::nn2_dist, J2*coupling::Heis);
    mc.define_coupling("J3a", pyrochlore::nn3a_dist, J3*coupling::Heis);
    mc.define_coupling("J3b", pyrochlore::nn3b_dist, J3*coupling::Heis);
    mc.set_global_field(global_field);

    mc.settings.T_ref = prog.get<double>("--T_ref");
    mc.setup_lattice();

    return mc;
}

// Initialise all spins to a spiral with wavevector Q = (0,0,Qz) (Qz in units of
// 2pi/a_cubic). Spins rotate in the yz-plane: S = (0, cos(phi), sin(phi)).
inline void init_spiral_state(CMC::Lattice& lat, double Qz_rounded) {
    constexpr double a_cubic = 8.;
    for (auto& s : lat.get_objects<CMC::HeisenbergSpin>()) {
        double phase = Qz_rounded * 2 * M_PI * static_cast<double>(s.ipos[2]) / a_cubic;
        s.S = {0., std::cos(phase), std::sin(phase)};
    }
}

inline auto name_LJ123(const argparse::ArgumentParser& prog){
    auto J1 = prog.get<double>("--J1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = resolve_J3(prog);
    auto Delta = prog.get<double>("--Delta");
    int L = prog.get<int>("L");

    std::stringstream name;
    name << "L="<<L<<DELIM<<
        "J1="<<J1<<DELIM<<
        "J2="<<J2<<DELIM<<
        "J3="<<J3<<DELIM<<
        "Delta="<<Delta<<DELIM;
    if (prog.is_used("--Qz")) {
        double Qz_rounded = round_Qz_to_supercell(prog.get<double>("--Qz"), L);
        name << "Qz="<<Qz_rounded<<DELIM;
    }
    return name.str();
}

// Generates a log-spaced array from T_hot to min(T_sample), making sure to
// include all of the T_sample steps exactly in sorted order.
// Returns also the 
inline std::pair<std::vector<double>, std::set<size_t>> generate_T_profile(
        double T_hot, double T_cold, const std::vector<double>& T_sample_unsorted, size_t total_steps){
    std::vector<double> T;
    std::set<size_t> idx;
    auto T_sample(T_sample_unsorted);
    // sort descending
    std::sort(T_sample.begin(),T_sample.end(), [](double a, double b){return a>b;});
   
    const double factor = pow(T_cold / T_hot, 1./total_steps);

    double t = T_hot/factor;
    size_t i_samp=0; // T_sample_copy[0] is biggest
                     //
    total_steps += T_sample.size();
    for (size_t i=0; i<total_steps; i++){
        if (t < T_sample[i_samp]){
            t=T_sample[i_samp]; 
            idx.insert(i);
            i_samp++;
        } else {
            t *= factor;
        }

        T.push_back(t);
    }

    return {T, idx};

}


