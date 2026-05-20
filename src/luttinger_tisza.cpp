#include <complex>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "H5Apublic.h"
#include "H5Gpublic.h"
#include "H5Ppublic.h"
#include "H5Tpublic.h"

#include "cli_bits.hpp"
#include "pyrochlore_geometry.hpp"
#include "format_bits.hpp"

// LatticeIndexing lives in supercell.hpp (included via cli_bits → pyrochlore_geometry → MC.hpp)
// We use it directly; no Supercell<HeisenbergSpin> is ever constructed.

using namespace std;

// Precomputed directed bond for the LT matrix.
// J = j·I_3 (scalar Heisenberg), so M(k) is n_sl×n_sl complex Hermitian.
// Each undirected bond appears twice (once per direction), so the physical
// energy per spin is (1/2)·λ_min.
struct LTBond {
    int    alpha, beta; // source / target sublattice (0..n_sl-1)
    ipos_t disp;        // full integer displacement from alpha to beta (= sl_pos[beta] + t_cell - sl_pos[alpha])
    double j;           // scalar exchange
};

static vector<LTBond> build_bonds(
        LatticeIndexing& lat,
        const vector<ipos_t>& sl_pos,
        double J1, double J2, double J3)
{
    using DistTable = const vector<vector<ipos_t>>;
    const struct { double j; DistTable* dist; } specs[] = {
        {J1, &pyrochlore::nn1_dist},
        {J2, &pyrochlore::nn2_dist},
        {J3, &pyrochlore::nn3a_dist},
        {J3, &pyrochlore::nn3b_dist},
    };

    const int n_sl = static_cast<int>(sl_pos.size());
    vector<LTBond> bonds;

    for (auto& [j, dist] : specs) {
        if (j == 0.0) continue;
        for (int sl = 0; sl < n_sl; sl++) {
            const int pyro_sl = sl % 4;
            for (const auto& v : (*dist)[pyro_sl]) {
                ipos_t ref = sl_pos[sl] + v; // copy; get_supercell_IDX mutates in place
                lat.get_supercell_IDX(ref);  // wrap ref to primitive cell (discard returned I)
                int beta = -1;
                for (int s = 0; s < n_sl; s++) {
                    if (sl_pos[s] == ref) { beta = s; break; }
                }
                assert(beta >= 0 && "bond target not found in sl_positions");
                bonds.push_back({sl, beta, v, j});
            }
        }
    }
    return bonds;
}


int main(int argc, char* argv[])
{
    argparse::ArgumentParser prog("ltgs");

    prog.add_argument("--output_dir", "-o")
        .help("Path to output directory");

    declare_LJ123(prog);

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err) {
        cerr << err.what() << "\n" << prog;
        return 1;
    }

    
    filesystem::path outdir;
    if (prog.is_used("--output_dir")){
        const string outdir_s = prog.get<string>("output_dir");
        outdir = outdir_s;
        if (!filesystem::exists(outdir))
            throw runtime_error("output_dir does not exist: " + outdir_s);
    }

    const double J1 = prog.get<double>("--J1");
    const double J2 = prog.get<double>("--J2");
    const double J3 = resolve_J3(prog);

    // Build LatticeIndexing directly — no Supercell<HeisenbergSpin> needed.
    const int L = prog.get<int>("L");
    const imat33_t prim_cell  = imat33_t::from_cols({8,0,0},{0,8,0},{0,0,8});
    const imat33_t supercell  = imat33_t::from_cols({L,0,0},{0,L,0},{0,0,L});
    LatticeIndexing lat(prim_cell, supercell);

    // Sublattice positions: 4 FCC sites × 4 pyrochlore link directions = 16 total.
    // Wrap each to the primitive cell so positions match what get_supercell_IDX returns.
    vector<ipos_t> sl_pos;
    sl_pos.reserve(16);
    for (const auto& fcc : pyrochlore::fcc_Dy) {
        for (const auto& x : pyrochlore::pyro) {
            ipos_t pos = x + fcc;
            lat.wrap_primitive(pos);
            sl_pos.push_back(pos);
        }
    }
    const int n_sl = static_cast<int>(sl_pos.size()); // = 16

    const int Nk           = lat.num_primitive_cells();
    const ivec3_t k_dims   = lat.size();
    const auto bonds       = build_bonds(lat, sl_pos, J1, J2, J3);

    // -----------------------------------------------------------------------
    // Sweep BZ: build M(k) and track minimum eigenvalue
    // -----------------------------------------------------------------------
    using MatC = Eigen::MatrixXcd;

    vector<double> eigenvalue_map(Nk);
    double E_min = numeric_limits<double>::max();
    idx3_t k_star_idx{};
    Eigen::VectorXcd eigvec_star(n_sl);

    for (int k_flat = 0; k_flat < Nk; k_flat++) {
        const idx3_t Q   = lat.idx3_from_flat(k_flat);
        const auto k_vec = lat.wavevector_from_idx3(Q);

        MatC M = MatC::Zero(n_sl, n_sl);
        for (const auto& b : bonds) {
            const double phase =
                k_vec[0] * (double)b.disp[0] +
                k_vec[1] * (double)b.disp[1] +
                k_vec[2] * (double)b.disp[2];
            M(b.alpha, b.beta) += b.j * complex<double>(cos(phase), sin(phase));
        }
        // M is Hermitian by construction (full directed-bond sweep)

        Eigen::SelfAdjointEigenSolver<MatC> eigs(M);
        const double lmin = eigs.eigenvalues()(0);
        eigenvalue_map[k_flat] = lmin;

        if (lmin < E_min) {
            E_min      = lmin;
            k_star_idx = Q;
            eigvec_star = eigs.eigenvectors().col(0);
        }
    }

    // Energy per spin: each undirected bond counted twice in M → factor of 1/2
    const double E_per_spin = 0.5 * E_min;

    const auto k_star_vec = lat.wavevector_from_idx3(k_star_idx);

    printf("LT minimum: λ_min = %.6f  E/spin = %.6f\n", E_min, E_per_spin);
    printf("Eigvec sublattice |s_α|² :");
    for (int s=0; s<n_sl; s++)
        printf(" %.4f", norm(eigvec_star(s)));
    printf("\n");

    printf("k*  idx = [%lld, %lld, %lld]\n",
           (long long)k_star_idx[0],
           (long long)k_star_idx[1],
           (long long)k_star_idx[2]);
    printf("k*  vec = [%.4f, %.4f, %.4f]  (rad / coord-unit)\n",
           k_star_vec[0], k_star_vec[1], k_star_vec[2]);


    // -----------------------------------------------------------------------
    // HDF5 output
    // -----------------------------------------------------------------------
    
    // exit if no output file provided
    if (! prog.is_used("--output_dir")) return 0;

    stringstream name;
    name << name_LJ123(prog);
    auto file_path = outdir / (name.str() + "ltgs.out.h5");

    hid_t fid = H5Fcreate(file_path.string().c_str(),
                           H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0)
        throw runtime_error("Failed to create " + file_path.string());

    auto write_1d = [&](const char* dname, hid_t type,
                        hsize_t len, const void* data) {
        hid_t sp = H5Screate_simple(1, &len, nullptr);
        hid_t ds = H5Dcreate2(fid, dname, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(sp);
    };

    // eigenvalue_map [Nk]
    write_1d("eigenvalue_map", H5T_NATIVE_DOUBLE,
             static_cast<hsize_t>(Nk), eigenvalue_map.data());

    // k_star_idx [3]
    {
        int64_t ksi[3] = { k_star_idx[0], k_star_idx[1], k_star_idx[2] };
        write_1d("k_star_idx", H5T_NATIVE_INT64, 3, ksi);
    }

    // k_star_vec [3]
    {
        double ksv[3] = { k_star_vec[0], k_star_vec[1], k_star_vec[2] };
        write_1d("k_star_vec", H5T_NATIVE_DOUBLE, 3, ksv);
    }

    // E_min scalar (energy per spin)
    {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t ds = H5Dcreate2(fid, "E_min", H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &E_per_spin);
        H5Dclose(ds);
        H5Sclose(sp);
    }

    // eigenvector [n_sl, 2] (columns: re, im)
    {
        vector<double> ev(2 * n_sl);
        for (int s = 0; s < n_sl; s++) {
            ev[2*s]   = eigvec_star(s).real();
            ev[2*s+1] = eigvec_star(s).imag();
        }
        hsize_t dims[2] = { static_cast<hsize_t>(n_sl), 2 };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(fid, "eigenvector", H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ev.data());
        H5Dclose(ds);
        H5Sclose(sp);
    }

    // /geometry group: lattice and reciprocal vectors
    {
        hid_t grp = H5Gcreate2(fid, "geometry",
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        auto write_mat = [&](const char* dname, hid_t type, const auto& matrix) {
            hsize_t dims[2] = {3, 3};
            decltype(matrix(0,0)) buf[9];
            for (int i = 0; i < 9; i++) buf[i] = matrix(i/3, i%3);
            hid_t sp = H5Screate_simple(2, dims, nullptr);
            hid_t ds = H5Dcreate2(grp, dname, type, sp,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
            H5Dclose(ds);
            H5Sclose(sp);
        };
        write_mat("index_cell",    H5T_NATIVE_INT64,  lat.get_lattice_vectors());
        write_mat("recip_vectors", H5T_NATIVE_DOUBLE, lat.get_reciprocal_lattice_vectors());

        H5Gclose(grp);
    }

    // k_dims attribute on root
    {
        hsize_t adim = 3;
        int kd[3] = { (int)k_dims[0], (int)k_dims[1], (int)k_dims[2] };
        hid_t sp = H5Screate_simple(1, &adim, nullptr);
        hid_t at = H5Acreate2(fid, "k_dims", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, kd);
        H5Aclose(at);
        H5Sclose(sp);
    }

    H5Fclose(fid);
    cout << "Saved to " << file_path << "\n";

    return 0;
}
