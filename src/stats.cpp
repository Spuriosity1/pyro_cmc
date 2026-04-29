#include "stats.hpp"

namespace CMC {

SublatWeightMatrix static_corr_3D::make_phase_mat(Lattice& lat) {
    const auto& sl_pos = std::get<SlPos<HeisenbergSpin>>(lat.sl_positions);
    return SublatWeightMatrix::phase_factors(
        lat.lattice,
        std::vector<ipos_t>(sl_pos.begin(), sl_pos.end()));
}

static_corr_3D::static_corr_3D(Lattice& lat_)
    : lat(lat_),
      ft_x(lat_),
      ft_y(lat_),
      ft_z(lat_),
      phase_mat(make_phase_mat(lat_))
{}


void static_corr_3D::declare_observable(const std::string& name,
                                        uint16_t correlators_to_count)
{
    if (correlators_to_count == 0)
        throw std::logic_error("Must request at least one correlator");
    observables.emplace_back(name, correlators_to_count,
                             static_cast<size_t>(n_k_points()));
    compute_correlators |= correlators_to_count;
}


void static_corr_3D::sample() {
    if (compute_correlators & (NEEDS_XX | NEEDS_XY | NEEDS_ZX)) ft_x.transform();
    if (compute_correlators & (NEEDS_XY | NEEDS_YY | NEEDS_ZY)) ft_y.transform();
    if (compute_correlators & (NEEDS_ZX | NEEDS_ZY | NEEDS_ZZ)) ft_z.transform();

    // Compute one correlator pair, contract with phase, accumulate into matching observables.
    auto accumulate_pair = [&](
            const FourierBuffer<HeisenbergSpin>& a,
            const FourierBuffer<HeisenbergSpin>& b,
            uint16_t pair_mask)
    {
        bool needed = false;
        for (const auto& obs : observables)
            if (obs.mask & pair_mask) { needed = true; break; }
        if (!needed) return;

        auto corr = correlate(a, b);
        auto contracted = phase_mat.contract(corr);

        for (auto& obs : observables) {
            if (!(obs.mask & pair_mask)) continue;
            for (int k = 0; k < n_k_points(); k++)
                obs.data[k] += contracted[k].real();
        }
    };

    accumulate_pair(ft_x.get_buffer(), ft_x.get_buffer(), NEEDS_XX);
    accumulate_pair(ft_y.get_buffer(), ft_y.get_buffer(), NEEDS_YY);
    accumulate_pair(ft_z.get_buffer(), ft_z.get_buffer(), NEEDS_ZZ);
    accumulate_pair(ft_x.get_buffer(), ft_y.get_buffer(), NEEDS_XY);
    accumulate_pair(ft_z.get_buffer(), ft_x.get_buffer(), NEEDS_ZX);
    accumulate_pair(ft_z.get_buffer(), ft_y.get_buffer(), NEEDS_ZY);

    n_samples++;
}


void static_corr_3D::write_K_points(hid_t file_id, const char* dset_name) {
    const int n0 = lat.lattice.size(0);
    const int n1 = lat.lattice.size(1);
    const int n2 = lat.lattice.size(2);

    hsize_t dims[4] = {(hsize_t)n0, (hsize_t)n1, (hsize_t)n2, 3};
    hid_t dataspace_id = H5Screate_simple(4, dims, nullptr);
    if (dataspace_id < 0)
        throw std::runtime_error("Failed to create dataspace");

    hid_t dataset_id = H5Dcreate2(file_id, dset_name, H5T_NATIVE_DOUBLE,
            dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        H5Sclose(dataspace_id);
        throw std::runtime_error("Failed to create dataset");
    }

    const auto B = lat.lattice.get_reciprocal_lattice_vectors();
    std::vector<double> k_points(3 * n0 * n1 * n2);

    const hsize_t s2 = 3;
    const hsize_t s1 = (hsize_t)n2 * s2;
    const hsize_t s0 = (hsize_t)n1 * s1;

    for (int i0 = 0; i0 < n0; i0++)
        for (int i1 = 0; i1 < n1; i1++)
            for (int i2 = 0; i2 < n2; i2++) {
                auto k = B * vector3::vec3d(i0, i1, i2);
                const hsize_t base = (hsize_t)i0*s0 + (hsize_t)i1*s1 + (hsize_t)i2*s2;
                k_points[base + 0] = k[0];
                k_points[base + 1] = k[1];
                k_points[base + 2] = k[2];
            }

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, k_points.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    if (status < 0)
        throw std::runtime_error("Failed to write k-points dataset");
}


void static_corr_3D::save_K_points(const std::filesystem::path& file_path) {
    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC,
            H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    write_K_points(file_id);
    H5Fclose(file_id);
}


void static_corr_3D::save(const std::filesystem::path& file_path) {
    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC,
            H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    write_group(file_id);
    H5Fclose(file_id);
}


void static_corr_3D::write_group(hid_t file_id, const char* group_name) {
    const int n0 = lat.lattice.size(0);
    const int n1 = lat.lattice.size(1);
    const int n2 = lat.lattice.size(2);

    hid_t data_group = H5Gcreate2(file_id, group_name,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (data_group < 0) {
        H5Fclose(file_id);
        throw std::runtime_error("Failed to create group");
    }

    {
        hsize_t dims[1] = {1};
        hid_t sp = H5Screate_simple(1, dims, NULL);
        hid_t ds = H5Dcreate2(data_group, "n_samples", H5T_NATIVE_INT,
                sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &n_samples);
        H5Dclose(ds);
        H5Sclose(sp);
    }

    for (const auto& obs : observables) {
        hsize_t dims[3] = {(hsize_t)n0, (hsize_t)n1, (hsize_t)n2};
        hid_t sp = H5Screate_simple(3, dims, NULL);
        hid_t ds = H5Dcreate2(data_group, obs.name.c_str(), H5T_NATIVE_DOUBLE,
                sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        herr_t status = H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, obs.data.data());
        H5Dclose(ds);
        H5Sclose(sp);
        if (status < 0) {
            H5Gclose(data_group);
            throw std::runtime_error("Failed to write dataset: " + obs.name);
        }
    }

    H5Gclose(data_group);
}


} // namespace CMC
