#include "stats.hpp"
#include "fftw3.h"

namespace CMC {
    void static_corr_3D::sample(){

        const int nk=n_k_points();
        const sl_t slcount = n_sublattices();

#ifndef NDEBUG
          // Basic sanity checks
    if ((int)ft_plans_X.size() != (int)slcount
     || (int)ft_plans_Y.size() != (int)slcount
     || (int)ft_plans_Z.size() != (int)slcount) {
        throw std::runtime_error("static_corr::sample: FT plans size mismatch with sublattice count");
    }
#endif

        // copies the appropriate data into place...
        std::vector<double> tmp_data;
        const auto& dummy = ft_plans_X[0];
        assert( nk == dummy.n0*dummy.n1*dummy.n2k);
        assert(nk == n_k_points());
        tmp_data.resize(nk);

        // Helper lambda to load and transform spin component data
        auto load_and_transform = [this](std::vector<FT_plan_3D>& plans, int spin_component) {
            for (sl_t sl = 0; sl < n_sublattices(); sl++) {
                const auto offset = lat.num_primitive * sl;
                for (int idx = 0; idx < lat.num_primitive; idx++) {
                    auto s = lat.links.at(idx + offset);
                    plans[sl].tmp[idx] = s->S[spin_component];
                }
                plans[sl].transform();
            }
        };
        
        // Helper lambda to compute correlators for a given mask
        auto compute_correlators_for_mask = [this, &tmp_data](
            const std::vector<FT_plan_3D>& plans1,
            const std::vector<FT_plan_3D>& plans2,
            uint32_t mask
        ) {
            for (sl_t sl1 = 0; sl1 < n_sublattices(); sl1++) {
                for (sl_t sl2 = 0; sl2 < n_sublattices(); sl2++) {
                    std::fill(tmp_data.begin(), tmp_data.end(), 0);
                    k_inner(plans1, sl1, plans2, sl2, tmp_data);
                    for (auto& o : observables) {
                        if (o.mask & mask) {
                            for (int jj=0; jj<n_k_points(); jj++){
                                o.data[jj] += tmp_data[jj];
                            }
                        }
                    }
                }
            }
        };
        
        if (compute_correlators & (NEEDS_XX | NEEDS_XY | NEEDS_ZX)) {
            load_and_transform(ft_plans_X, 0);
        }

        if (compute_correlators & (NEEDS_XY | NEEDS_YY | NEEDS_ZY)) {
            load_and_transform(ft_plans_Y, 1);
        }

        if (compute_correlators & (NEEDS_ZX | NEEDS_ZY | NEEDS_ZZ)) {
            load_and_transform(ft_plans_Z, 2);
        }

        compute_correlators_for_mask(ft_plans_X, ft_plans_X, NEEDS_XX);
        compute_correlators_for_mask(ft_plans_Y, ft_plans_Y, NEEDS_YY);
        compute_correlators_for_mask(ft_plans_Z, ft_plans_Z, NEEDS_ZZ);

        compute_correlators_for_mask(ft_plans_X, ft_plans_Y, NEEDS_XY);
        compute_correlators_for_mask(ft_plans_Z, ft_plans_Y, NEEDS_ZY);
        compute_correlators_for_mask(ft_plans_Z, ft_plans_X, NEEDS_ZX);
    
        n_samples++;
    }


    void static_corr_3D::save_K_points(const std::filesystem::path& file_path) {
        // Create HDF5 file
        hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
        }

        write_K_points(file_id);

        // Cleanup
        H5Fclose(file_id);
    }

    void static_corr_3D::write_K_points(hid_t file_id, const char* dset_name){
        // Write BZ_points as (n_primitive, 3) dataset
        try {
            // dimension: n0, n1, n2k, 3
            hsize_t n0 = lat.size(0);
            hsize_t n1 = lat.size(1);
            hsize_t n2k = lat.size(2)/2 + 1;

            hsize_t dims[4] = {n0, n1, n2k, 3};
            hid_t dataspace_id = H5Screate_simple(4, dims, nullptr);
            if (dataspace_id < 0) {
                throw std::runtime_error("Failed to create dataspace");
            }

            hid_t dataset_id = H5Dcreate2(file_id, dset_name, H5T_NATIVE_DOUBLE, 
                    dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (dataset_id < 0) {
                H5Sclose(dataspace_id);
                throw std::runtime_error("Failed to create dataset");
            }

            // Prepare k-points data
            std::vector<double> k_points(3 * n0 * n1 * n2k);
            // each index [m0, m1, m2] corresponds to k-point 
            // k = 2pi * [m0/d0, m1/d1, m2/d2] * a^-1
            auto ainv = vector3::mat33<double>::from_other(
                    lat.primitive_spec.latvecs_unnormed_inverse
                    );
            ainv *= 1.0/lat.primitive_spec.abs_det_latvecs;

            hsize_t s2 = dims[3];            // 3
            hsize_t s1 = dims[2] * s2;       // n2k * 3
            hsize_t s0 = dims[1] * s1;       // n1 * n2k * 3
            
            for (hsize_t i0=0; i0<n0; i0++){
                for (hsize_t i1=0; i1<n1; i1++){
                    for (hsize_t i2=0; i2<n2k; i2++){
                        auto k = 2*M_PI*vector3::vec3d(
                                1.*i0/lat.size(0),
                                1.*i1/lat.size(1),
                                1.*i2/lat.size(2))* ainv;
                        k_points[i0*s0 + i1*s1 + i2*s2 + 0] = k[0];
                        k_points[i0*s0 + i1*s1 + i2*s2 + 1] = k[1];
                        k_points[i0*s0 + i1*s1 + i2*s2 + 2] = k[2];
                    }
                }
            }
        

            herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                    H5P_DEFAULT, k_points.data());

            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);

            if (status < 0) {
                throw std::runtime_error("Failed to write k-points dataset");
            }
        }
        catch (...) {
            H5Fclose(file_id);
            throw;  // Re-throw the exception after cleanup
        }

    }

    void static_corr_3D::save(const std::filesystem::path& file_path) {
        
        // Create HDF5 file
        hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
        }

        write_group(file_id);
        
        H5Fclose(file_id);
    }

    void static_corr_3D::write_group(hid_t file_id, const char* group_name){

        hid_t data_group = H5Gcreate2(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (data_group < 0) {
            H5Fclose(file_id);
            throw std::runtime_error("Failed to create /ssf group");
        }
        

        // Write n_samples as single integer
        {
            hsize_t dims[1] = {1};
            hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
            
            hid_t dataset_id = H5Dcreate2(data_group, "n_samples", H5T_NATIVE_INT, 
                                          dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            int n_samp = n_samples;  
            herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, 
                                     H5P_DEFAULT, &n_samp);
            
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            
            if (status < 0) {
                H5Gclose(data_group);
                H5Fclose(file_id);
                throw std::runtime_error("Failed to write n_samples dataset");
            }
        }
        
        // Write each observable's data
        for (const auto& obs : observables) {
//            hsize_t dims[1] = {static_cast<hsize_t>(obs.data.size())};
//            hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
            hsize_t dims[3] = {
                static_cast<hsize_t>(lat.size(0)),
                static_cast<hsize_t>(lat.size(1)),
                static_cast<hsize_t>(lat.size(2)/2 + 1)
            };
            hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
            
            hid_t dataset_id = H5Dcreate2(data_group, obs.name.c_str(), H5T_NATIVE_DOUBLE, 
                                          dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                                     H5P_DEFAULT, obs.data.data());
            
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            
            if (status < 0) {
                H5Gclose(data_group);
                H5Fclose(file_id);
                throw std::runtime_error("Failed to write dataset: " + obs.name);
            }
        }
        
        // Close groups and file
        H5Gclose(data_group);
    }


} // namespace CMC
