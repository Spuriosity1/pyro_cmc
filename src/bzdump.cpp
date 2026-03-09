#include "MC.hpp"
#include "argparse.hpp"
#include "stats.hpp"
#include <iostream>
#include "format_bits.hpp"
#include "preset_cellspecs.hpp"


template<typename T>
requires std::convertible_to<T, double>
void write_mat33d(hid_t file_id, const char* dset_name, vector3::mat33<T> mat)
{
    hsize_t dims[2] = {3,3};
    hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);
    if (dataspace_id < 0)
        throw std::runtime_error("Failed to create dataspace");

    hid_t dataset_id = H5Dcreate2(file_id, dset_name, H5T_NATIVE_DOUBLE, 
            dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        H5Sclose(dataspace_id);
        throw std::runtime_error("Failed to create dataset");
    }

    vector3::mat33<double> M = vector3::mat33<double>::from_other(mat);

    double Mdata[9] = {
        M(0,0), M(0,1), M(0,2),
        M(1,0), M(1,1), M(1,2),
        M(2,0), M(2,1), M(2,2)
    };

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
            H5P_DEFAULT, Mdata);

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    if (status < 0) {
        throw std::runtime_error("Failed to write k-points dataset");
    }
}


using namespace std;
int main (int argc, char *argv[]) {
    ///////////////////////////////////////////////////////////////////////////
    /// Setup and CLI
    argparse::ArgumentParser prog(argv[0]);

    /// BOOK-KEEPING
    prog.add_argument("--output_dir", "-o")
        .help("Path to output")
        .required();


    /// PHYSICAL
    prog.add_argument("L")
        .help("Linear dimension of cubic supercell (e.g. L=2 has 2^3 *4 = 32 primitive cells)")
        .scan<'i', int>(); // cubic unit cell hardcoded

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
        cerr << err.what() << endl;
        cerr << prog;
        std::exit(1);
    }


    /// Ensuring directories exist AHEAD of time (avoids heartbreak)
    std::string outdir_s = prog.get<std::string>("output_dir");
    filesystem::path outdir(outdir_s);
    if (! filesystem::exists(outdir) ){
        throw runtime_error("Cannot open outdir");
    }

    int L = prog.get<int>("L");
    auto supercell_spec = imat33_t::from_cols( {-L, L, L}, {L, -L, L}, {L, L, -L});

    const auto spec = CellGeometry::PrimitiveSpecifiers::DiamondSpec();
    CMC::Lattice lat(spec, supercell_spec); 
    CMC::static_corr_3D ssf_manager(lat);

    stringstream name;
    name<<"L="<<L<<DELIM;
    auto f = outdir /( name.str() + ".BZ.h5" );
    printf("Saving BZ data to %s\n", f.string().c_str());

    

    hid_t file_id = H5Fcreate(f.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + f.string());
    }

    ssf_manager.write_K_points(file_id, "k_points");
    write_mat33d(file_id, "cell_vectors", lat.cell_vectors);
    write_mat33d(file_id, "index_cell_vectors", lat.index_cell_vectors);
    write_mat33d(file_id, "primitive_vectors", lat.primitive_spec.latvecs);
    write_mat33d(file_id, "inverse_primitive", 
            1.0/lat.primitive_spec.abs_det_latvecs * 
            vector3::mat33<double>::from_other(lat.primitive_spec.latvecs_unnormed_inverse));






    // Cleanup
    H5Fclose(file_id);


    

    
}
