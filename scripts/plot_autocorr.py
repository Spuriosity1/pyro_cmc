import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("file", help="filename")
ap.add_argument("--window", default=1, type=int)
ap.add_argument("-T", default=None, help="T index", type=int)

args = ap.parse_args()


f = h5py.File(args.file, 'r')

g = f['/autocorr']

T_list = np.array(g['T_list'])
acorr = g['autocorr']

cmap = plt.colormaps['viridis']
norm = mpl.colors.Normalize(vmin=min(np.log(T_list)), vmax=max(np.log(T_list)))

def bin(arr, window):
    N = len(arr) // window
    X = arr[0:(N*window)].reshape(window, N)
    return np.mean(X, axis=0), np.sqrt(np.var(X, axis=0))

if args.T is None:
    for ti, T in enumerate(T_list):
        acorr_slice = np.array(acorr[ti, :])
        y, dy = bin(acorr_slice, args.window)
        plt.errorbar(np.arange(len(y))*args.window, y, yerr=dy,
                 color=cmap(norm(np.log(T))), label=f"T={T}")

    plt.legend()
else:
    ti = args.T
    acorr_slice = np.array(acorr[ti, :])
    y, dy = bin(acorr_slice, args.window)
    plt.errorbar(np.arange(len(y))*args.window, y, yerr=dy,
             color=cmap(norm(np.log(T))), label=f"T={T}")

plt.axhline(0, color='r')
plt.show()
