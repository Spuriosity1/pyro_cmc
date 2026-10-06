#include <complex>
#include <cstdint>
#include <limits>
#include <random>
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
// For XXZ, J is a 3×3 matrix, so M(k) is (3·n_sl)×(3·n_sl) complex Hermitian.
// Each undirected bond appears twice (once per direction), so the physical
// energy per spin is (1/2)·λ_min.
struct LTBond {
    int    alpha, beta;      // source / target sublattice (0..n_sl-1)
    ipos_t disp;             // full integer displacement from alpha to beta
    Eigen::Matrix3d J;       // 3×3 coupling matrix
};

static vector<LTBond> build_bonds(
        LatticeIndexing& lat,
        const vector<ipos_t>& sl_pos,
        double J1, double J2, double J3, double J4, double Jzz)
{
    // Coupling shells keyed by squared spin separation (cubic cell side = 8):
    // J1 -> 8, J2 -> 24, J3 -> 32, J4 -> 40. J1 keeps its hand table because the
    // XXZ frame is indexed per sublattice pair; the further shells come from the
    // separation generator (matches build_J1J2J3_h, and folds the old nn3a/nn3b
    // split into the single dist_sq=32 shell). {strength, table, xxz-applies}.
    struct Spec { double j; vector<vector<ipos_t>> dist; bool xxz; };
    const Spec specs[] = {
        {J1, pyrochlore::nn1_dist,                   true},
        {J2, pyrochlore::neighbours_by_distance(24), false},
        {J3, pyrochlore::neighbours_by_distance(32), false},
        {J4, pyrochlore::neighbours_by_distance(40), false},
    };

    const int n_sl = static_cast<int>(sl_pos.size());
    vector<LTBond> bonds;

    for (auto& [j, dist, xxz] : specs) {
        if (j == 0.0) continue;
        for (int sl = 0; sl < n_sl; sl++) {
            const int pyro_sl = sl % 4;
            for (const auto& v : dist[pyro_sl]) {
                ipos_t ref = sl_pos[sl] + v;
                lat.get_supercell_IDX(ref);
                int beta = -1;
                for (int s = 0; s < n_sl; s++) {
                    if (sl_pos[s] == ref) { beta = s; break; }
                }
                assert(beta >= 0 && "bond target not found in sl_positions");

                Eigen::Matrix3d J_mat;
                if (xxz && Jzz != 0.0) {
                    // J_bond = j·[I + (Delta-1)·ẑ_μ ⊗ ẑ_ν]
                    // where μ = pyro_sl of source, ν = pyro_sl of target
                    int mu = sl   % 4;
                    int nu = beta % 4;
                    const auto& zm = pyrochlore::axis[mu][2];
                    const auto& zn = pyrochlore::axis[nu][2];
                    Eigen::Vector3d z_mu(zm[0], zm[1], zm[2]);
                    Eigen::Vector3d z_nu(zn[0], zn[1], zn[2]);
                    J_mat = j * Eigen::Matrix3d::Identity()
                                 + Jzz * z_mu * z_nu.transpose();
                } else {
                    J_mat = j * Eigen::Matrix3d::Identity();
                }
                bonds.push_back({sl, beta, v, J_mat});
            }
        }
    }
    return bonds;
}

// -----------------------------------------------------------------------
// Continuous LT band-minimum optimiser
//
// M(k) is (3·n_sl)² complex Hermitian, M[3α:3α+3, 3β:3β+3] = Σ J_bond·e^{ik·d}.
// The LT ground state is the k minimising the lowest eigenvalue λ_min(k). We
// find it by gradient descent instead of scanning the L³ commensurate k-grid.
//
//   M is analytic in k, so by Hellmann–Feynman
//     ∂λ_min/∂k_i = v† (∂M/∂k_i) v ,     ∂M/∂k_i block = Σ J_bond·(i d_i)·e^{ik·d}
//   with v the (normalised) lowest eigenvector. ∂M/∂k_i is Hermitian, so the
//   quadratic form is real: per bond, s = v_α† (J·e^{ik·d}) v_β contributes
//   Re(i d_i s) = −d_i·Im(s) to the i-th gradient component.
// -----------------------------------------------------------------------
using MatC = Eigen::MatrixXcd;

struct LTSolution {
    double           lambda;   // lowest eigenvalue of M(k)
    Eigen::VectorXcd v;        // corresponding (normalised) eigenvector
    Eigen::Vector3d  k;        // wavevector this was evaluated at
};

static MatC build_M(const vector<LTBond>& bonds,
                    const Eigen::Vector3d& k, int M_dim) {
    MatC M = MatC::Zero(M_dim, M_dim);
    for (const auto& b : bonds) {
        const double phase =
            k[0]*(double)b.disp[0] + k[1]*(double)b.disp[1] + k[2]*(double)b.disp[2];
        const complex<double> eph(cos(phase), sin(phase));
        M.block<3,3>(3*b.alpha, 3*b.beta) += b.J.cast<complex<double>>() * eph;
    }
    return M;
}

static LTSolution solve_min(const vector<LTBond>& bonds,
                            const Eigen::Vector3d& k, int M_dim) {
    Eigen::SelfAdjointEigenSolver<MatC> eigs(build_M(bonds, k, M_dim));
    return { eigs.eigenvalues()(0), eigs.eigenvectors().col(0), k };
}

static Eigen::Vector3d lt_gradient(const vector<LTBond>& bonds,
                                   const Eigen::Vector3d& k,
                                   const Eigen::VectorXcd& v) {
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (const auto& b : bonds) {
        const double phase =
            k[0]*(double)b.disp[0] + k[1]*(double)b.disp[1] + k[2]*(double)b.disp[2];
        const complex<double> eph(cos(phase), sin(phase));
        const Eigen::Vector3cd va = v.segment<3>(3*b.alpha);
        const Eigen::Vector3cd vb = v.segment<3>(3*b.beta);
        // s = v_α† (J·e^{ik·d}) v_β ; Vector3cd::dot conjugates its first arg.
        const complex<double> s = va.dot((b.J.cast<complex<double>>() * vb) * eph);
        for (int i = 0; i < 3; i++) g[i] -= (double)b.disp[i] * s.imag();
    }
    return g;
}

// Gradient descent with Armijo backtracking from a single seed k. Returns the
// converged solution (lowest λ reached from this basin).
static LTSolution descend(const vector<LTBond>& bonds, Eigen::Vector3d k,
                          int M_dim, int max_iter, double gtol) {
    LTSolution sol = solve_min(bonds, k, M_dim);
    double step = 1.0;                     // line-search seed, grows on success
    for (int it = 0; it < max_iter; it++) {
        const Eigen::Vector3d g = lt_gradient(bonds, sol.k, sol.v);
        const double gnorm = g.norm();
        if (gnorm < gtol) break;

        double eta = step;
        bool moved = false;
        for (int bt = 0; bt < 50; bt++) {
            const LTSolution s_try = solve_min(bonds, sol.k - eta * g, M_dim);
            if (s_try.lambda <= sol.lambda - 1e-4 * eta * gnorm * gnorm) {
                sol = s_try; moved = true;
                step = 2.0 * eta;          // be optimistic next iteration
                break;
            }
            eta *= 0.5;
        }
        if (!moved) break;                 // stuck at a minimum
    }
    return sol;
}


int main(int argc, char* argv[])
{
    argparse::ArgumentParser prog("ltgs");

    prog.add_argument("--output_dir", "-o")
        .help("Path to output directory");

    provide_physical_args(prog);

    // Optimiser controls (gradient descent over the continuous wavevector).
    prog.add_argument("--restarts")
        .help("Number of gradient-descent restarts (random seeds + Gamma)")
        .default_value(48)
        .scan<'i', int>();
    prog.add_argument("--max_iter")
        .help("Max gradient-descent iterations per restart")
        .default_value(500)
        .scan<'i', int>();
    prog.add_argument("--gtol")
        .help("Gradient-norm convergence tolerance")
        .default_value(1e-9)
        .scan<'g', double>();
    prog.add_argument("--opt_seed")
        .help("RNG seed for restart wavevectors")
        .default_value(0)
        .scan<'i', int>();

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

    const double J1    = prog.get<double>("--J1");
    const double J2    = prog.get<double>("--J2");
    const double J3    = resolve_J3(prog);
    const double J4    = prog.get<double>("--J4");
    const double Jzz = prog.get<double>("--Jzz");

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

    const ivec3_t k_dims   = lat.size();
    const auto bonds       = build_bonds(lat, sl_pos, J1, J2, J3, J4, Jzz);

    // -----------------------------------------------------------------------
    // Minimise λ_min(k) by gradient descent (see descend()/lt_gradient()).
    //
    // λ_min(k) is periodic under the primitive reciprocal lattice, so the
    // continuous optimum lives in one primitive BZ. We seed one descent at Γ
    // plus (restarts−1) random points spanning that BZ and keep the best basin.
    // The BZ is B·[−L/2, L/2)³ where B = supercell reciprocal vectors (the
    // L³-grid spacing) — i.e. exactly the k reached by the old commensurate
    // sweep, but now optimised off-grid.
    // -----------------------------------------------------------------------
    const int M_dim = 3 * n_sl;

    const int    n_restart = prog.get<int>("--restarts");
    const int    max_iter  = prog.get<int>("--max_iter");
    const double gtol      = prog.get<double>("--gtol");
    const uint64_t opt_seed = (uint64_t)prog.get<int>("--opt_seed");

    Eigen::Matrix3d B;   // supercell reciprocal lattice vectors (index -> k)
    {
        const auto R = lat.get_reciprocal_lattice_vectors();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) B(i, j) = R(i, j);
    }
    const Eigen::Matrix3d Binv = B.inverse();

    std::mt19937_64 rng(opt_seed);
    std::uniform_real_distribution<double> unif(-0.5, 0.5);

    LTSolution best{ numeric_limits<double>::max(), Eigen::VectorXcd::Zero(M_dim),
                     Eigen::Vector3d::Zero() };
    for (int r = 0; r < n_restart; r++) {
        Eigen::Vector3d k = Eigen::Vector3d::Zero();   // r == 0 -> Γ seed
        if (r > 0) {
            const Eigen::Vector3d frac(L*unif(rng), L*unif(rng), L*unif(rng));
            k = B * frac;
        }
        const LTSolution sol = descend(bonds, k, M_dim, max_iter, gtol);
        if (sol.lambda < best.lambda) best = sol;
    }

    // Energy per spin: each undirected bond counted twice in M → factor of 1/2
    const double E_min      = best.lambda;
    const double E_per_spin = 0.5 * E_min;
    const Eigen::VectorXcd& eigvec_star = best.v;

    // Fold the optimal k back into the first BZ and report the nearest
    // supercell-commensurate index for reference / HDF5 compatibility.
    Eigen::Vector3d frac = Binv * best.k;              // continuous supercell index
    for (int i = 0; i < 3; i++) frac[i] -= L * std::round(frac[i] / L);
    const Eigen::Vector3d k_star_vec = B * frac;

    idx3_t k_star_idx{};
    for (int i = 0; i < 3; i++) {
        long q = std::lround(frac[i]);
        k_star_idx[i] = ((q % L) + L) % L;
    }

    printf("LT minimum: λ_min = %.6f  E/spin = %.6f  (%d restarts)\n",
           E_min, E_per_spin, n_restart);
    // Print per-sublattice weight: sum of |components|² over the 3 spin dofs
    printf("Eigvec sublattice |s_α|² :");
    for (int s = 0; s < n_sl; s++) {
        double w = norm(eigvec_star(3*s)) + norm(eigvec_star(3*s+1)) + norm(eigvec_star(3*s+2));
        printf(" %.4f", w);
    }
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

    // eigenvector [3*n_sl, 2] (re/im; rows ordered as [spin-x,y,z for sl-0, sl-1, ...])
    {
        vector<double> ev(2 * M_dim);
        for (int s = 0; s < M_dim; s++) {
            ev[2*s]   = eigvec_star(s).real();
            ev[2*s+1] = eigvec_star(s).imag();
        }
        hsize_t dims[2] = { static_cast<hsize_t>(M_dim), 2 };
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
