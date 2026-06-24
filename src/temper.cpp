#include <algorithm>
#include <cmath>
#include <memory>
#include <mpi.h>

#include "MC.hpp"
#include "cli_bits.hpp"
#include "ssf_manager.hpp"
#include "format_bits.hpp"

/*
Parallel tempering via MPI: one MPI rank per replica.  Each rank holds its
spin configuration permanently; when two adjacent-rung replicas exchange, we
swap their S-vector arrays over MPI rather than copying replica objects.

Protocol per round:
  1. n_sweep local Metropolis sweeps at the rank's fixed temperature T.
  2. Sampled ranks (those in S_idx) run the SSF FFT; non-sampled ranks do one
     extra sweep — these happen in parallel across MPI processes.
  3. Compute local energy (used for both the heat-capacity accumulator and the
     exchange decision).
  4. Checkerboard exchange: even rounds pair ranks (0,1),(2,3),...; odd rounds
     pair (1,2),(3,4),....  The lower (hotter) rank of each pair makes the
     Metropolis decision and sends it to the upper (cooler) rank; if accepted
     both sides exchange their S-vector buffers.

Output: a single .out.h5 file written by rank 0 containing /ssf, /energy, and
/geometry groups, with the SSF accumulated at every T in --T_sample and energy
accumulated at every rung.
*/

using namespace std;
using namespace CMC;

// ---------------------------------------------------------------------------
// Exchange S vectors between this rank and `partner` over MPI.
// ---------------------------------------------------------------------------
static void exchange_spins(Lattice& lat, int partner)
{
    auto& spins = lat.get_objects<HeisenbergSpin>();
    const int n = static_cast<int>(spins.size());
    std::vector<double> send_buf(3 * n), recv_buf(3 * n);
    for (int i = 0; i < n; i++) {
        send_buf[3*i]   = spins[i].S[0];
        send_buf[3*i+1] = spins[i].S[1];
        send_buf[3*i+2] = spins[i].S[2];
    }
    MPI_Sendrecv(send_buf.data(), 3*n, MPI_DOUBLE, partner, 10,
                 recv_buf.data(), 3*n, MPI_DOUBLE, partner, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n; i++) {
        spins[i].S[0] = recv_buf[3*i];
        spins[i].S[1] = recv_buf[3*i+1];
        spins[i].S[2] = recv_buf[3*i+2];
    }
}

// ---------------------------------------------------------------------------
// Write /energy group directly to an open HDF5 file.
// T_list, E, E2, n_samples are all length n (one entry per rung), sorted
// ascending in T.
// ---------------------------------------------------------------------------
static void write_energy_group(hid_t fid,
                                const std::vector<double>& T_list,
                                const std::vector<double>& E,
                                const std::vector<double>& E2,
                                const std::vector<size_t>& n_samples)
{
    const hsize_t len = T_list.size();
    hid_t grp = H5Gcreate2(fid, "/energy",
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t sp  = H5Screate_simple(1, &len, nullptr);

    auto write1d = [&](const char* name, hid_t type, const void* data) {
        hid_t ds = H5Dcreate2(grp, name, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
    };
    write1d("T_list",    H5T_NATIVE_DOUBLE,   T_list.data());
    write1d("E",         H5T_NATIVE_DOUBLE,   E.data());
    write1d("E2",        H5T_NATIVE_DOUBLE,   E2.data());
    write1d("n_samples", H5T_NATIVE_ULLONG,   n_samples.data());

    H5Sclose(sp);
    H5Gclose(grp);
}

// ---------------------------------------------------------------------------
// Write /ssf group to an open HDF5 file from pre-gathered arrays.
//
// ssf_T_list:  temperatures, in the order of ssf_blocks (one per sampled rung)
// ssf_blocks:  flat [n_corr, n_k, n_sl, n_sl, 2] per temperature
// ssf_sq_blocks: same for squared-correlator
// ssf_n_samples: n_samples per temperature
// The function sorts output ascending in T, matching plot_ssf.py expectations.
// ---------------------------------------------------------------------------
static void write_ssf_group(hid_t fid,
                             const std::vector<double>& ssf_T_list,
                             const std::vector<std::vector<double>>& ssf_blocks,
                             const std::vector<std::vector<double>>& ssf_sq_blocks,
                             const std::vector<size_t>& ssf_n_samples,
                             const std::vector<std::string>& corr_labels,
                             const std::vector<ipos_t>& sl_positions,
                             int n_sl, int n_kpoints, int n_spins, ivec3_t k_dims)
{
    const size_t n_T    = ssf_T_list.size();
    const size_t n_corr = corr_labels.size();
    const size_t nk     = static_cast<size_t>(n_kpoints);
    const size_t ns     = static_cast<size_t>(n_sl);

    // Sort ascending in T.
    std::vector<size_t> ord(n_T);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](size_t a, size_t b){ return ssf_T_list[a] < ssf_T_list[b]; });

    hid_t grp = H5Gcreate2(fid, "/ssf",
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // --- static_corr / static_corr_2  [n_corr, n_T, n_k, n_sl, n_sl, 2] ---
    {
        hsize_t dims[6] = { n_corr, n_T, nk, ns, ns, 2 };
        hid_t sp = H5Screate_simple(6, dims, nullptr);

        const size_t n_flat = n_corr * nk * ns * ns * 2;
        std::vector<double> full(n_corr * n_T * n_flat);
        std::vector<double> full_sq(n_corr * n_T * n_flat);

        for (size_t ti = 0; ti < n_T; ti++) {
            const auto& src    = ssf_blocks   [ord[ti]];
            const auto& src_sq = ssf_sq_blocks[ord[ti]];
            // Remap per-T flat [n_corr, n_k, n_sl, n_sl, 2] →
            //         full    [n_corr, n_T, n_k, n_sl, n_sl, 2]
            for (size_t c = 0; c < n_corr; c++) {
                const size_t slice     = nk * ns * ns * 2;
                const size_t base_src  = c * slice;
                const size_t base_full = (c * n_T + ti) * slice;
                std::copy(src.begin()    + base_src,
                          src.begin()    + base_src + slice,
                          full.begin()   + base_full);
                std::copy(src_sq.begin() + base_src,
                          src_sq.begin() + base_src + slice,
                          full_sq.begin()+ base_full);
            }
        }

        hid_t ds = H5Dcreate2(grp, "static_corr", H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, full.data());
        H5Dclose(ds);

        hid_t ds2 = H5Dcreate2(grp, "static_corr_2", H5T_NATIVE_DOUBLE, sp,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, full_sq.data());
        H5Dclose(ds2);

        H5Sclose(sp);
    }

    // --- corr_lookup ---
    {
        hid_t str_t = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_t, H5T_VARIABLE);
        H5Tset_cset(str_t, H5T_CSET_UTF8);
        hsize_t nc = n_corr;
        hid_t sp = H5Screate_simple(1, &nc, nullptr);
        hid_t ds = H5Dcreate2(grp, "corr_lookup", str_t, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        std::vector<const char*> ptrs(n_corr);
        for (size_t c = 0; c < n_corr; c++) ptrs[c] = corr_labels[c].c_str();
        H5Dwrite(ds, str_t, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data());
        H5Dclose(ds);
        H5Sclose(sp);
        H5Tclose(str_t);
    }

    // --- T_list and n_samples (sorted ascending) ---
    {
        hsize_t len = n_T;
        hid_t sp = H5Screate_simple(1, &len, nullptr);

        std::vector<double> sT(n_T);
        std::vector<size_t> sN(n_T);
        for (size_t i = 0; i < n_T; i++) {
            sT[i] = ssf_T_list   [ord[i]];
            sN[i] = ssf_n_samples[ord[i]];
        }

        hid_t ds_T = H5Dcreate2(grp, "T_list",    H5T_NATIVE_DOUBLE, sp,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds_T, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, sT.data());
        H5Dclose(ds_T);

        hid_t ds_N = H5Dcreate2(grp, "n_samples", H5T_NATIVE_ULLONG, sp,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds_N, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, sN.data());
        H5Dclose(ds_N);

        H5Sclose(sp);
    }

    // --- sl_positions [n_sl, 3] ---
    {
        hsize_t dims[2] = { ns, 3 };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, "sl_positions", H5T_NATIVE_INT64, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 sl_positions.data());
        H5Dclose(ds);
        H5Sclose(sp);
    }

    // --- attributes: n_spins, k_dims ---
    {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t at = H5Acreate2(grp, "n_spins", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, &n_spins);
        H5Aclose(at);
        H5Sclose(sp);
    }
    {
        hsize_t adim = 3;
        int kd[3] = { static_cast<int>(k_dims[0]),
                      static_cast<int>(k_dims[1]),
                      static_cast<int>(k_dims[2]) };
        hid_t sp = H5Screate_simple(1, &adim, nullptr);
        hid_t at = H5Acreate2(grp, "k_dims", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, kd);
        H5Aclose(at);
        H5Sclose(sp);
    }

    H5Gclose(grp);
}


// ===========================================================================
int main(int argc, char* argv[])
{
    MPI_Init(nullptr, nullptr);

    int replica_id, n_replicas;
    MPI_Comm_rank(MPI_COMM_WORLD, &replica_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_replicas);

    ///////////////////////////////////////////////////////////////////////////
    // CLI
    argparse::ArgumentParser prog("temper");

    prog.add_argument("--output_dir", "-o").help("Path to output").required();
    prog.add_argument("--seed", "-s")
        .required()
        .help("Base seed index (2*n_replicas consecutive indices consumed)")
        .scan<'i', size_t>();
    prog.add_argument("--save_state")
        .help("Save the spin state of the coldest replica")
        .implicit_value(true).default_value(false);
    prog.add_argument("--init_spiral")
        .help("Pre-initialise spins to a spiral (requires --Q)")
        .implicit_value(true).default_value(false);

    prog.add_argument("--T_hot")
        .help("Temperature of the hottest replica")
        .scan<'g', double>();
    prog.add_argument("--T_cold")
        .help("Temperature of the coldest replica (defaults to min(T_sample))")
        .scan<'g', double>();
    prog.add_argument("--T_ref")
        .help("Reference temperature for proposal distribution")
        .default_value(1.0).scan<'g', double>();
    prog.add_argument("--T_sample")
        .help("Temperatures at which to accumulate the SSF")
        .nargs(argparse::nargs_pattern::at_least_one)
        .required().scan<'g', double>();
    prog.add_argument("--n_steps")
        .help("Total number of replicas (including T_sample rungs)")
        .default_value(static_cast<size_t>(100)).scan<'i', size_t>();
    prog.add_argument("--n_sweep", "-w")
        .help("Local-update sweeps per replica between exchange attempts")
        .default_value(static_cast<size_t>(16)).scan<'i', size_t>();
    prog.add_argument("--n_burn_in")
        .help("Sweeps at fixed T before exchange attempts begin")
        .default_value(static_cast<size_t>(16)).scan<'i', size_t>();
    prog.add_argument("--n_sample")
        .default_value(static_cast<size_t>(64))
        .help("Number of exchange rounds while collecting statistics")
        .scan<'i', size_t>();
    prog.add_argument("--prefix").default_value("run");

    declare_LJ123(prog);

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err) {
        if (replica_id == 0) { cerr << err.what() << "\n" << prog; }
        MPI_Finalize();
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Input validation
    std::string outdir_s = prog.get<std::string>("output_dir");
    filesystem::path outdir(outdir_s);
    if (!filesystem::exists(outdir)) {
        if (replica_id == 0)
            fprintf(stderr, "ERROR: output_dir does not exist: %s\n", outdir_s.c_str());
        MPI_Finalize();
        return 1;
    }

    const size_t seed      = prog.get<size_t>("--seed");
    const double T_hot     = prog.is_used("--T_hot") ? prog.get<double>("--T_hot") : 10.0;
    const std::vector<double> T_sample = prog.get<std::vector<double>>("--T_sample");
    const double T_cold    = prog.is_used("--T_cold") ?
        prog.get<double>("--T_cold") :
        *std::min_element(T_sample.begin(), T_sample.end());
    const size_t n_sweep   = prog.get<size_t>("--n_sweep");
    const size_t n_burn_in = prog.get<size_t>("--n_burn_in");
    const size_t n_sample  = prog.get<size_t>("--n_sample");

    if (n_replicas <= static_cast<int>(T_sample.size())) {
        if (replica_id == 0)
            fprintf(stderr, "ERROR: n_replicas (%d) must exceed T_sample count (%zu)\n",
                    n_replicas, T_sample.size());
        MPI_Finalize();
        return 1;
    }
    if (n_replicas < 2) {
        if (replica_id == 0)
            fprintf(stderr, "ERROR: need at least 2 replicas\n");
        MPI_Finalize();
        return 1;
    }

    auto [T_grid, S_idx] = generate_T_profile(
        T_hot, T_cold, T_sample,
        static_cast<size_t>(n_replicas) - T_sample.size());

    const double T = T_grid[static_cast<size_t>(replica_id)];
    const bool is_sampled = (S_idx.count(static_cast<size_t>(replica_id)) > 0);

    ///////////////////////////////////////////////////////////////////////////
    // Lattice + MC construction.  Each rank uses a unique seed offset so that
    // replicas start with independent spin configurations.
    auto lat = build_pyro_lat(prog);
    auto mc  = build_J1J2J3_h(prog, lat, seed + static_cast<size_t>(replica_id));

    if (prog.get<bool>("--init_spiral")) {
        if (!prog.is_used("--Q")) {
            if (replica_id == 0)
                fprintf(stderr, "ERROR: --init_spiral requires --Q\n");
            MPI_Finalize();
            return 1;
        }
        double Q_rounded = round_Qz_to_supercell(
            prog.get<double>("--Q"), prog.get<int>("L"));
        int spiral_axis = prog.get<int>("--spiral_axis");
        init_spiral_state(lat, Q_rounded, spiral_axis);
    }

    auto B = prog.get<std::vector<double>>("-B");
    if (B.size() < 3) {
        if (replica_id == 0)
            fprintf(stderr, "ERROR: --B requires 3 arguments\n");
        MPI_Finalize();
        return 1;
    }

    std::stringstream name;
    name << prog.get<std::string>("--prefix") << DELIM << name_LJ123(prog)
         << "B=" << B[0] << "," << B[1] << "," << B[2] << DELIM
         << "seed=" << seed << DELIM
         << "Tc=" << T_cold << DELIM
         << "sw=" << n_sweep << DELIM;

    ///////////////////////////////////////////////////////////////////////////
    // Burn-in: each rank sweeps independently at its own temperature.
    if (replica_id == 0)
        printf("[temper] Burning in %d replicas (%zu sweeps each)...\n",
               n_replicas, n_burn_in);

    for (size_t i = 0; i < n_burn_in; i++)
        mc.sweep_local_Metropolis(T);

    ///////////////////////////////////////////////////////////////////////////
    // SSF manager: only sampled ranks allocate one.
    std::unique_ptr<ssf_manager> ssfm_ptr;
    if (is_sampled) {
        ssfm_ptr = std::make_unique<ssf_manager>(lat, std::vector<std::string>{"xx","yy","zz"}, 1, true);
        ssfm_ptr->new_T(T);
    }

    // Per-rank energy accumulators.
    double E_sum = 0.0, E2_sum = 0.0;
    size_t n_e   = 0;

    // Total swap statistics tracked by lower-index partner of each pair.
    size_t pair_attempts = 0, pair_accepts = 0;

    // Swap RNG: independent per rank (lower partners are the ones that use it).
    XoshiroCpp::Xoroshiro128PlusPlus swap_rng(
        hash_true_random(seed + static_cast<size_t>(n_replicas)
                              + static_cast<size_t>(replica_id)));
    std::uniform_real_distribution<double> rand01(0.0, 1.0);

    const double Np = static_cast<double>(lat.lattice.num_primitive_cells());

    if (replica_id == 0)
        printf("[temper] Begin sampling (%d rungs, %zu sampled, %zu rounds)...\n",
               n_replicas, S_idx.size(), n_sample);

    ///////////////////////////////////////////////////////////////////////////
    // Main loop
    for (size_t round = 0; round < n_sample; round++) {

        // 1. Local sweeps at this rank's temperature.
        for (size_t sw = 0; sw < n_sweep; sw++)
            mc.sweep_local_Metropolis(T);

        // 2. SSF sample (sampled ranks) OR extra sweep (non-sampled ranks).
        //    These execute in parallel across MPI processes.
        if (is_sampled)
            ssfm_ptr->sample();
        else
            mc.sweep_local_Metropolis(T);

        // 3. Sample energy (after the last config change this round).
        const double e = mc.total_energy_per_unit_cell();
        E_sum  += e;
        E2_sum += e * e;
        n_e++;

        // 4. Checkerboard exchange.
        //    Even rounds pair (0,1),(2,3),...; odd rounds pair (1,2),(3,4),...
        const size_t start = round % 2;
        const size_t rid   = static_cast<size_t>(replica_id);

        const bool is_lower = (rid % 2 == start) && (rid + 1 < static_cast<size_t>(n_replicas));
        const bool is_upper = (rid % 2 != start) && (rid > 0);
        const int partner   = is_lower ? replica_id + 1
                            : is_upper ? replica_id - 1
                            : -1;

        if (partner >= 0) {
            // Exchange energies (total, not per-cell).
            double my_E_total      = e * Np;
            double partner_E_total = 0.0;
            MPI_Sendrecv(&my_E_total,      1, MPI_DOUBLE, partner, 1,
                         &partner_E_total, 1, MPI_DOUBLE, partner, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int accept = 0;
            if (is_lower) {
                // Lower rank is hotter: T_self > T_partner → 1/T_self < 1/T_partner
                const double T_partner = T_grid[static_cast<size_t>(partner)];
                const double arg = (1.0/T - 1.0/T_partner)
                                   * (my_E_total - partner_E_total);
                accept = (arg >= 0.0 || rand01(swap_rng) < std::exp(arg)) ? 1 : 0;
                MPI_Send(&accept, 1, MPI_INT, partner, 2, MPI_COMM_WORLD);
                pair_attempts++;
                if (accept) pair_accepts++;
            } else {
                MPI_Recv(&accept, 1, MPI_INT, partner, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (accept)
                exchange_spins(lat, partner);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Gather swap statistics to rank 0 and print summary.
    {
        // Each rank is the "lower partner" for exactly the pair (rank, rank+1),
        // and tracks pair_attempts / pair_accepts for that pair.
        std::vector<size_t> all_attempts(n_replicas, 0), all_accepts(n_replicas, 0);
        MPI_Gather(&pair_attempts, 1, MPI_UNSIGNED_LONG,
                   all_attempts.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&pair_accepts,  1, MPI_UNSIGNED_LONG,
                   all_accepts.data(),  1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        if (replica_id == 0) {
            printf("\n[temper] per-pair swap acceptance:\n");
            for (int a = 0; a + 1 < n_replicas; a++) {
                const size_t att = all_attempts[a];
                const size_t acc = all_accepts [a];
                printf("  pair %3d<->%3d  T=%.4e <-> %.4e  accept=%5.1f%%  (%zu/%zu)\n",
                       a, a+1,
                       T_grid[static_cast<size_t>(a)],
                       T_grid[static_cast<size_t>(a)+1],
                       att ? 100.0 * acc / att : 0.0,
                       acc, att);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Gather energy data (one scalar per rank) to rank 0.
    std::vector<double> all_E(n_replicas, 0.0), all_E2(n_replicas, 0.0);
    std::vector<size_t> all_n(n_replicas, 0);
    MPI_Gather(&E_sum,  1, MPI_DOUBLE,        all_E.data(),  1, MPI_DOUBLE,        0, MPI_COMM_WORLD);
    MPI_Gather(&E2_sum, 1, MPI_DOUBLE,        all_E2.data(), 1, MPI_DOUBLE,        0, MPI_COMM_WORLD);
    MPI_Gather(&n_e,    1, MPI_UNSIGNED_LONG, all_n.data(),  1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    ///////////////////////////////////////////////////////////////////////////
    // Gather SSF data from sampled ranks to rank 0.
    // Rank 0 knows S_idx (computed identically on every rank) so it knows
    // exactly which ranks to recv from.
    //
    // Buffer layout per sampled rank: [n_corr, n_k, n_sl, n_sl, 2]
    // (same size for all because all ranks share the same lattice geometry).

    // Compute buffer size from geometry (valid on all ranks without ssf_manager).
    const int n_sl_val = static_cast<int>(
        std::get<SlPos<HeisenbergSpin>>(lat.sl_positions).size());
    const int n_kp_val = lat.lattice.num_primitive_cells();
    const int n_corr_val = 3; // "xx","yy","zz"
    const size_t ssf_buf_size =
        static_cast<size_t>(n_corr_val) * static_cast<size_t>(n_kp_val)
        * static_cast<size_t>(n_sl_val) * static_cast<size_t>(n_sl_val) * 2;

    // Sampled ranks: serialize and send.
    std::vector<double> my_corr_buf, my_corr_sq_buf;
    size_t my_ssf_n = 0;
    if (is_sampled) {
        ssfm_ptr->get_flat_buffer(0, my_corr_buf, my_corr_sq_buf);
        my_ssf_n = ssfm_ptr->get_n_samples_at(0);
        if (replica_id != 0) {
            MPI_Send(my_corr_buf.data(),    ssf_buf_size, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD);
            MPI_Send(my_corr_sq_buf.data(), ssf_buf_size, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);
            MPI_Send(&my_ssf_n, 1, MPI_UNSIGNED_LONG, 0, 22, MPI_COMM_WORLD);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Rank 0: assemble and write HDF5.
    if (replica_id == 0) {
        // Collect SSF from sampled ranks.
        const size_t n_sampled = S_idx.size();
        std::vector<double>              ssf_T_list(n_sampled);
        std::vector<std::vector<double>> ssf_blocks(n_sampled),
                                         ssf_sq_blocks(n_sampled);
        std::vector<size_t>              ssf_n_samp(n_sampled);

        size_t slot = 0;
        for (size_t idx : S_idx) {
            ssf_T_list[slot] = T_grid[idx];
            if (static_cast<int>(idx) == 0) {
                // Rank 0 itself is a sampled rank.
                ssf_blocks   [slot] = my_corr_buf;
                ssf_sq_blocks[slot] = my_corr_sq_buf;
                ssf_n_samp   [slot] = my_ssf_n;
            } else {
                ssf_blocks   [slot].resize(ssf_buf_size);
                ssf_sq_blocks[slot].resize(ssf_buf_size);
                MPI_Recv(ssf_blocks   [slot].data(), ssf_buf_size, MPI_DOUBLE,
                         static_cast<int>(idx), 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(ssf_sq_blocks[slot].data(), ssf_buf_size, MPI_DOUBLE,
                         static_cast<int>(idx), 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&ssf_n_samp[slot], 1, MPI_UNSIGNED_LONG,
                         static_cast<int>(idx), 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            slot++;
        }

        // Sort energy output ascending in T (T_grid is descending: rank 0 = hottest).
        std::vector<double> e_T(n_replicas), e_E(n_replicas), e_E2(n_replicas);
        std::vector<size_t> e_n(n_replicas);
        for (int k = 0; k < n_replicas; k++) {
            int asc = n_replicas - 1 - k; // ascending-T index
            e_T  [asc] = T_grid[static_cast<size_t>(k)];
            e_E  [asc] = all_E [k];
            e_E2 [asc] = all_E2[k];
            e_n  [asc] = all_n [k];
        }

        // SSF metadata from rank 0's lattice.
        const auto& sl_pos_var = lat.sl_positions;
        const auto& sl_pos_vec = std::get<SlPos<HeisenbergSpin>>(sl_pos_var);
        std::vector<ipos_t> sl_positions(sl_pos_vec.begin(), sl_pos_vec.end());
        const int n_spins_val = static_cast<int>(
            lat.get_objects<HeisenbergSpin>().size());
        const ivec3_t k_dims_val = lat.lattice.size();

        const std::vector<std::string> corr_labels = {"xx", "yy", "zz"};

        // Write HDF5.
        auto file_path = outdir / (name.str() + ".out.h5");
        hid_t file_id  = H5Fcreate(file_path.string().c_str(),
                                    H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0)
            throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());

        write_energy_group(file_id, e_T, e_E, e_E2, e_n);
        write_ssf_group(file_id, ssf_T_list, ssf_blocks, ssf_sq_blocks,
                        ssf_n_samp, corr_labels, sl_positions,
                        n_sl_val, n_kp_val, n_spins_val, k_dims_val);
        write_geometry_group(file_id, lat);

        H5Fclose(file_id);
        printf("[temper] Saved to %s\n", file_path.string().c_str());
    }

    ///////////////////////////////////////////////////////////////////////////
    // Save coldest replica's spin state (rank n_replicas-1, always the coldest
    // since T_grid is descending).
    if (prog.get<bool>("--save_state") && replica_id == n_replicas - 1) {
        auto f = outdir / (name.str() + ".spins.h5");
        printf("[temper] Saving spin state of coldest replica (T=%.3e) to %s\n",
               T, f.string().c_str());
        save_spin_state(lat, f);
    }

    MPI_Finalize();
    return 0;
}
