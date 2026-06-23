#pragma once



#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "abstract_manager.hpp"
#include <cassert>
#include <filesystem>
#include <vector>


// Per-temperature energy accumulator. T_list/n_samples/new_T/set_T are
// inherited from abstract_manager: set_T() lets a single instance accumulate
// samples for several interleaved temperatures (e.g. parallel tempering
// slots that get visited in any order across rounds), not just one
// sequentially-advancing temperature.
class energy_manager : public abstract_manager {
    std::vector<double> E;
    std::vector<double> E2;

    void on_new_temp() override {
        E.push_back(0);
        E2.push_back(0);
    }

    public:
    energy_manager(size_t n_temperatures_reserve=0) {
        E.reserve(n_temperatures_reserve);
        E2.reserve(n_temperatures_reserve);
    }

    double curr_E() const {
        return E[curr_idx] / n_samples[curr_idx];
    }

    void sample(double _e){
        assert(!T_list.empty());

        E[curr_idx] += _e;
        E2[curr_idx] += (_e*_e);
        n_samples[curr_idx]++;
    }

    void save(const std::filesystem::path& file_path);
    void write_group(hid_t file_id, const char* group_name="/energy") override;
};



inline void energy_manager::save(const std::filesystem::path& file_path){

    // Create HDF5 file
    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }
    write_group(file_id);

    // Close groups and file
    H5Fclose(file_id);
}

inline void energy_manager::write_group(hid_t file_id, const char* group_name){
    hid_t data_group = H5Gcreate2(file_id, group_name,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (data_group < 0) {
        throw std::runtime_error("Failed to create group");
    }

    // All vectors must have the same length
    const hsize_t dims[1] = { E.size() };
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    if (dataspace_id < 0) {
        H5Gclose(data_group);
        throw std::runtime_error("Failed to create dataspace");
    }

    auto write_dataset = [&](const char* name,
            hid_t type,
            const void* data)
    {
        hid_t dataset_id = H5Dcreate2(data_group, name, type,
                dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id < 0) {
            H5Sclose(dataspace_id);
            H5Gclose(data_group);
            throw std::runtime_error(std::string("Failed to create dataset ") + name);
        }

        herr_t status = H5Dwrite(dataset_id, type,
                H5S_ALL, H5S_ALL,
                H5P_DEFAULT, data);

        H5Dclose(dataset_id);

        if (status < 0) {
            H5Sclose(dataspace_id);
            H5Gclose(data_group);
            throw std::runtime_error(std::string("Failed to write dataset ") + name);
        }
    };

    // Write all datasets
    write_dataset("E",       H5T_NATIVE_DOUBLE, E.data());
    write_dataset("E2",      H5T_NATIVE_DOUBLE, E2.data());
    write_dataset("T_list",  H5T_NATIVE_DOUBLE, T_list.data());
    write_dataset("n_samples", H5T_NATIVE_ULLONG, n_samples.data());

    // Cleanup
    H5Sclose(dataspace_id);
    H5Gclose(data_group);
}
