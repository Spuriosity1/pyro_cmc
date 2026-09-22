#pragma once

#include "H5Fpublic.h"
#include "H5Ppublic.h"
#include <stdexcept>
#include <string>

// Create (truncating) an HDF5 file with POSIX advisory file locking DISABLED.
//
// HDF5 1.10+ takes an flock() on every file it opens. On networked/parallel
// cluster filesystems (Lustre, NFS, GPFS) that lock request intermittently
// fails with errno 11 (EAGAIN, "Resource temporarily unavailable") when many
// jobs hammer the lock manager at once, aborting the run at H5Fcreate. Each of
// our outputs is written by exactly one process, so the lock protects nothing
// — turn it off. Equivalent to exporting HDF5_USE_FILE_LOCKING=FALSE, but baked
// into the binary so it can't be forgotten in a job script.
inline hid_t h5_make_fapl_nolock() {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0)
        throw std::runtime_error("Failed to create HDF5 file-access plist");
    // use_file_locking = false, ignore_disabled_locking = true.
    H5Pset_file_locking(fapl, false, true);
    return fapl;
}

inline hid_t h5_create_trunc_nolock(const std::string& path) {
    hid_t fapl = h5_make_fapl_nolock();
    hid_t file_id = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + path);
    return file_id;
}

// Reopen an existing HDF5 file read-write, again with locking disabled. Used to
// briefly reattach for each streamed per-temperature write and for the final
// metadata write, so the file is never held open during the long MC sweeps.
inline hid_t h5_open_rdwr_nolock(const std::string& path) {
    hid_t fapl = h5_make_fapl_nolock();
    hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDWR, fapl);
    H5Pclose(fapl);
    if (file_id < 0)
        throw std::runtime_error("Failed to open HDF5 file: " + path);
    return file_id;
}
