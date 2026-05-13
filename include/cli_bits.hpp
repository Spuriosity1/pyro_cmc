#pragma once
#include "format_bits.hpp"
#include "vec3.hpp"
#include <argparse.hpp>
#include <filesystem>
#include <iostream>
#include <ostream>

#include "vec3.hpp"
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
        .help("Third-nearest-neighbour Heisenberg coupling strength")
        .default_value(0.)
        .scan<'g', double>();
    prog.add_argument("--external_field", "-B")
        .help("Global magnetic field")
        .nargs(3)
        .default_value(std::vector<double>({0.,0.,0.}))
        .scan<'g', double>();
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

inline auto build_J1J2J3(const argparse::ArgumentParser& prog, CMC::Lattice& lat, uint64_t seed){

    vector3::vec3d global_field;
    { 
        auto B_tmp = prog.get<std::vector<double>>("-B");
        for (int i=0; i<3; i++){ global_field[i]= B_tmp[i]; }
    }

    auto J1 = prog.get<double>("--J1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = prog.get<double>("--J3");

    CMC::MC_runner mc(lat, seed);
    mc.define_coupling("J1", pyrochlore::nn1_dist, J1*coupling::Heis);
    mc.define_coupling("J2", pyrochlore::nn2_dist, J2*coupling::Heis);
    mc.define_coupling("J3a", pyrochlore::nn3a_dist, J3*coupling::Heis);
    mc.define_coupling("J3b", pyrochlore::nn3b_dist, J3*coupling::Heis);
    mc.define_global_field(global_field);

    mc.settings.T_ref = prog.get<double>("--T_ref");
    mc.setup_lattice();

    return mc;
}

inline auto name_LJ123(const argparse::ArgumentParser& prog){
    auto J1 = prog.get<double>("--J1");
    auto J2 = prog.get<double>("--J2");
    auto J3 = prog.get<double>("--J3");
    int L = prog.get<int>("L");

    std::stringstream name; // accumulates hashed options
    name <<"L="<<L<<DELIM<<
        "J1="<<J1<<DELIM<<
        "J2="<<J2<<DELIM<<
        "J3="<<J3<<DELIM;
    return name.str();
}



