#include "MC.hpp"
#include <random>

namespace CMC {



    void MC_runner::setup_lattice(){
        int idx=0;
        for (const auto& c : coupling_specs){
            std::cout<<"Coupling Index "<<idx++<<" -> linking\n";
            for (auto [link_id, link]: lat.links){
                sl_t sl = lat.primitive_spec.sl_of_link(link->position);

                std::vector<HeisenbergSpin*> shell_above;
                std::vector<HeisenbergSpin*> shell_below;
                for (const auto& v : c.relative_vectors.at(sl)){
                    HeisenbergSpin* other = &lat.get_link_at(link->position + v);
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




    void MC_runner::define_coupling(const std::string& name, 
            const std::vector<std::vector<idx3_t>>& rel_vecs, 
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


    // helper functions
    //
    void accumulate_field(vector3::vec3d& h, 
            const std::vector<HeisenbergSpin*> spin_list){
        for (auto& s : spin_list) {h += s->S;}
    }

    double norm(const vector3::vec3d& v){
        return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    vector3::vec3d MC_runner::local_field(const HeisenbergSpin *spin
            ) const 
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
            h_loc += tmp * J ; // left-multiply: effectively transposes J
        }
        return h_loc;
    }

    void mirror_about_vector(vector3::vec3d& v, const vector3::vec3d& axis){

        // S :> S - 2*(S - (h_loc dot S) * h_loc / |h_loc|^2)
        //     = -S + 2* h_loc (h_loc dot S ) / h_loc ^2
        v = -v + 2 *(dot(axis, v) / (dot(axis, axis) + 1e-10) ) * axis;
    }
    
    size_t MC_runner::local_Metropolis(double T, HeisenbergSpin* spin)
    {
        auto h_loc = local_field(spin) + global_field;
        double curr_E = dot(spin->S, h_loc);
        // simplest update I can imagine: add a random Gaussian; renormalise
        // control the Gaussian strength with T to avoid doing something stupid
 

        // sometimes, mirror across the field direction
        // (zero-energy move, always accepted)
        if (rand01(rng) <0.5){
            mirror_about_vector(spin->S, h_loc);
        }
    
        // propose step
        auto new_S = sqrt(T/settings.T_ref) * vector3::vec3d(
                normal_dist(rng), normal_dist(rng), normal_dist(rng)); 
        new_S += spin->S;
        new_S /= norm(new_S);

        double new_E = dot(new_S, h_loc);

        // Metropolis acceptance criterion
        double dE = new_E - curr_E;
        if (dE < 0 || rand01(rng) < exp(-dE / T)) {
            spin->S = new_S;
            return 1;  // accepted
        }
        
        return 0;  // rejected
    }

    size_t MC_runner::sweep_local_Metropolis(double T){
        size_t accepted = 0;
        for (auto [id, spin] : lat.links){
            accepted += local_Metropolis(T, spin);
        }
        return accepted;
    }

    double MC_runner::total_energy_per_unit_cell() const{
        double E=0;
        for (const auto& [il, s] : lat.links){
            E += 0.5 * dot( s->S,  local_field(s));
            E += dot(s->S, global_field);
        }
        return E / lat.num_primitive;
        // can be sped up 2x by keeping a de-duplicated bond list
        // do this if it's a bottleneck
    }

    
    void save_spin_state(const Lattice& lat, const std::filesystem::path& file_path){
        // Count number of spins
        const size_t N = lat.links.size();

        // Flatten buffers (N × 3)
        std::vector<int32_t> pos(3 * N);
        std::vector<double> ori(3 * N);

        size_t idx=0;
        for (const auto& [li, s] : lat.links ) {
            for (int i=0; i<3; i++){
                pos[3*idx+i] = s->position[i];
                ori[3*idx+i] = s->S[i];
            }
            ++idx;
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


} // end namespace



