#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <lattice_IO.hpp>
#include <argparse.hpp>



[[maybe_unused]] static const char* DELIM="?";

template<typename T>
inline std::string comma_separate(const char* pname, std::vector<T> v){
    std::ostringstream oss;
    oss<<pname<<"=";
    for (size_t i=0; i<v.size(); i++){
        oss << v[i];
        if (i < v.size() -1 ) oss << ",";
    }
    oss << ";";
    return oss.str();
}


inline std::string parse_supercell_spec(imat33_t& supercell_spec, 
        const argparse::ArgumentParser& prog){


    auto Z1 = prog.get<std::vector<int>>("Z1");
    auto Z2 = prog.get<std::vector<int>>("Z2");
    auto Z3 = prog.get<std::vector<int>>("Z3");


    for (int row=0; row<3; row++){
        supercell_spec(row,0) = Z1[row];
        supercell_spec(row,1) = Z2[row];
        supercell_spec(row,2) = Z3[row];
    }

    return comma_separate("Z1", Z1)
         + comma_separate("Z2", Z2)
         + comma_separate("Z3", Z3);
}
