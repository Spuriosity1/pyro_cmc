#include "MC.hpp"
#include "H5Gpublic.h"
#include <random>
#include <cmath>

namespace CMC {


    void MC_runner::setup_lattice(){
        int coup_idx = 0;
        for (const auto& c : coupling_specs){
            std::cout << "Coupling Index " << coup_idx++ << " -> linking\n";

            auto& spins = lat.get_objects<HeisenbergSpin>();
            const int Np = lat.lattice.num_primitive_cells();
            const int num_sl = static_cast<int>(
                std::get<SlPos<HeisenbergSpin>>(lat.sl_positions).size());

            for (int sl = 0; sl < num_sl; sl++) {
                // pyrochlore sublattice 0-3 determined by position within FCC site
                const int pyro_sl = sl % 4;

                for (int cell = 0; cell < Np; cell++) {
                    HeisenbergSpin* link = &spins[sl * Np + cell];

                    std::vector<HeisenbergSpin*> shell_above;
                    std::vector<HeisenbergSpin*> shell_below;

                    for (const auto& v : c.relative_vectors.at(pyro_sl)) {
                        HeisenbergSpin* other =
                            lat.get_object_at<HeisenbergSpin>(link->ipos + v);
                        if (other < link) {
                            shell_above.push_back(other);
                        } else {
                            shell_below.push_back(other);
                        }
                    }
                    link->bond_sets.push_back({shell_above, shell_below});
                }
            }
        }
    }



    void MC_runner::define_coupling(const std::string& name,
            const std::vector<std::vector<ipos_t>>& rel_vecs,
            const vector3::mat33<double>& J)
    {
        if (index.contains(name)){
            throw std::logic_error("Coupling names must be unique");
        }

        index[name] = coupling_specs.size();
        coupling_specs.push_back({name, rel_vecs, J});
    }


    void MC_runner::define_global_field(const vector3::vec3<double>& h){
        global_field = h;
    }


    void accumulate_field(vector3::vec3d& h,
            const std::vector<HeisenbergSpin*> spin_list){
        for (auto& s : spin_list) {h += s->S;}
    }

    double norm(const vector3::vec3d& v){
        return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    vector3::vec3d MC_runner::local_field(const HeisenbergSpin *spin) const
    {
        vector3::vec3d h_loc{0,0,0};
        vector3::vec3d tmp;
        assert(coupling_specs.size() == spin->bond_sets.size());
        for (size_t cpl_idx = 0; cpl_idx < coupling_specs.size(); cpl_idx++) {
            auto J = coupling_specs[cpl_idx].J;
            tmp = {0,0,0};
            accumulate_field(tmp, spin->bond_sets[cpl_idx].bonds_above);
            h_loc += J * tmp;
            tmp = {0,0,0};
            accumulate_field(tmp, spin->bond_sets[cpl_idx].bonds_below);
            h_loc += tmp * J;
        }
        return h_loc;
    }

    void mirror_about_vector(vector3::vec3d& v, const vector3::vec3d& axis){
        v = -v + 2 *(dot(axis, v) / (dot(axis, axis) + 1e-10) ) * axis;
    }

    size_t MC_runner::local_Metropolis(double T, HeisenbergSpin* spin)
    {
        auto h_loc = local_field(spin) + global_field;
        double curr_E = dot(spin->S, h_loc);

        if (rand01(rng) < 0.5){
            mirror_about_vector(spin->S, h_loc);
        }

        auto new_S = sqrt(T/settings.T_ref) * vector3::vec3d(
                normal_dist(rng), normal_dist(rng), normal_dist(rng));
        new_S += spin->S;
        new_S /= norm(new_S);

        double new_E = dot(new_S, h_loc);

        double dE = new_E - curr_E;
        if (dE < 0 || rand01(rng) < exp(-dE / T)) {
            spin->S = new_S;
            return 1;
        }

        return 0;
    }

    size_t MC_runner::sweep_local_Metropolis(double T){
        size_t accepted = 0;
        for (auto& spin : lat.get_objects<HeisenbergSpin>()){
            accepted += local_Metropolis(T, &spin);
        }
        return accepted;
    }

    double MC_runner::total_energy_per_unit_cell() const{
        double E = 0;
        for (const auto& s : std::get<std::vector<HeisenbergSpin>>(lat.objects)){
            E += 0.5 * dot(s.S, local_field(&s));
            E += dot(s.S, global_field);
        }
        return E / lat.lattice.num_primitive_cells();
    }


    void save_spin_state(const Lattice& lat, const std::filesystem::path& file_path){
        const auto& spins = std::get<std::vector<HeisenbergSpin>>(lat.objects);
        const size_t N = spins.size();

        std::vector<int32_t> pos(3 * N);
        std::vector<double> ori(3 * N);

        for (size_t idx = 0; idx < N; idx++) {
            const auto& s = spins[idx];
            for (int i = 0; i < 3; i++){
                pos[3*idx+i] = static_cast<int32_t>(s.ipos[i]);
                ori[3*idx+i] = s.S[i];
            }
        }

        hid_t file = H5Fcreate(file_path.string().c_str(),
                H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        hsize_t dims[2] = {N, 3};
        hid_t space = H5Screate_simple(2, dims, NULL);

        hid_t dset_pos = H5Dcreate(file, "spin_pos", H5T_STD_I32LE, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_pos, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, pos.data());
        H5Dclose(dset_pos);

        hid_t dset_ori =
            H5Dcreate(file, "spin_orientation", H5T_IEEE_F64LE, space,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_ori, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 ori.data());
        H5Dclose(dset_ori);

        H5Sclose(space);
        H5Fclose(file);
    }



// Write a "/geometry" group to an open HDF5 file with lattice statistics
// that are fixed for this disorder realisation:
//
//   n_spins            (scalar)  — number of non-deleted spins
//   n_tetras_by_intact (hsize_t[5]) — n_tetras_by_intact[k] = number of
//                                     tetrahedra with exactly k intact spins
 void write_geometry_group(hid_t file_id, Lattice& sc,
                                 const char* group_name) {

    hid_t grp = H5Gcreate2(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//    hid_t scalar_space = H5Screate(H5S_SCALAR);
//    hid_t ds = H5Dcreate2(grp, "n_spins", H5T_NATIVE_HSIZE,
//                           scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//    H5Dwrite(ds, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &n_spins);
//    H5Dclose(ds);
//    H5Sclose(scalar_space);


    auto write_mat = [&](const char* name, const auto& matrix, hid_t type) {
        hsize_t dims[2] = { 3, 3 };
        decltype(matrix(0,0)) data_rm[9];
        for (int i=0; i<9; i++){
            data_rm[i] = matrix(i/3, i%3);
        }
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, name, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) {
            H5Sclose(sp);
            H5Gclose(grp);
            throw std::runtime_error(std::string("ssf_manager: failed to create ") + name);
        }
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_rm);
        H5Dclose(ds);
        H5Sclose(sp);

    };


    write_mat("index_cell", sc.lattice.get_lattice_vectors(), H5T_NATIVE_INT64);
    write_mat("recip_vectors", sc.lattice.get_reciprocal_lattice_vectors(), H5T_NATIVE_DOUBLE);

    H5Gclose(grp);
}


} // end namespace
