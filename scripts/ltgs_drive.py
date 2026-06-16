import subprocess
import re
import numpy as np


pattern = r"k\*\s+vec\s*=\s*\[([^\]]+)\]"

tol = 1e-10

L = 48

J3_arr = np.arange(0.0,1.0,0.01)
k_arr = []

for J3 in J3_arr:
    print(J3)
    cmd=['build/ltgs', str(L), '--J1', '-1', '--J2', '-0.15', '--J3', str(J3)]
    res =subprocess.run(cmd, stdout=subprocess.PIPE)

    Qmin = np.nan
    match = re.search(pattern, res.stdout.decode('utf8'))
    if match:
        k_star = [float(x) for x in match.group(1).split(",")]
        for triple in ((0,1,2), (1,2,0), (2,0,1)):
            ix, iy, iz = triple
            if np.abs(k_star[ix]) > tol:
                if np.abs(k_star[iy]) < tol and np.abs(k_star[iz]) < tol:
                    Qmin = np.max(np.abs(k_star))
            else:
                Qmin = np.max(np.abs(k_star))
    k_arr.append(Qmin)

np.save("J3_arr.npy", J3_arr)
np.save("min_q.npy", k_arr)
