import h5py

def load_file(path):
    with h5py.File(path, "r") as f:
        recip = f["/geometry/recip_vectors"][:]   # (3, 3), rows are a*, b*, c*
        index_cell = f["/geometry/index_cell"][:]  # (3, 3)

        T_list = f["/energy/T_list"][:]
        E      = f["/energy/E"][:]
        E2     = f["/energy/E2"][:]
        n_E    = f["/energy/n_samples"][:]

        SdotS    = f["/ssf/SdotS"][:]     # (n_T, Nk0, Nk1)
        SzSz     = f["/ssf/SzSz"][:]
        n_ssf    = f["/ssf/n_samples"][:]
    return recip, index_cell, T_list, E, E2, n_E, SdotS, SzSz, n_ssf
