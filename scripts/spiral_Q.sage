#!/usr/bin/env sage
# ---------------------------------------------------------------------------
# Spiral (0,0,Q) wavevector from the Luttinger-Tisza / single-Q spiral energy
# on the pyrochlore lattice, for couplings J1..J4.
#
# Ansatz (identical to init_spiral_state / spiral_byhand): every site carries
#   S_i = ( 0, cos(q_z z_i), sin(q_z z_i) ),   q_z = Q * 2*pi / 8   (a_cubic = 8)
# so a bond of displacement d contributes J * cos(q_z * d_z). The energy per
# spin is therefore a sum over neighbour shells of J_shell * (z-projection sum).
#
# CONVENTION: nearest-neighbour J1 is ferromagnetic, J1 = -1 (matches the code:
# spiral_byhand defaults --J1=-1 and resolve_J3 rescales by |J1|). This is the
# sign that makes the result reproduce the existing cli_bits::J3_from_Qz.
# ---------------------------------------------------------------------------
from itertools import product

# --- neighbour shells by squared separation (same rule as neighbours_by_distance)
R_fcc = [(0,0,0),(0,4,4),(4,0,4),(4,4,0)]
link  = [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]
def shell_dz(d2):
    """Multiset of z-displacements over one spin's d2-shell (averaged per site)."""
    nc = int(sqrt(d2)/8) + 1
    dzs = []
    for pr in range(4):
        ref = vector(link[pr])
        for c in product(range(-nc,nc+1), repeat=3):
            for f in R_fcc:
                for l in link:
                    d = vector(f)+vector(l)+8*vector(c)-ref
                    if d.dot_product(d) == d2:
                        dzs.append(d[2])
    return dzs   # length = 4 * (bonds per site)

var('q J2 J3 J4')
J1 = -1                                   # FM convention (see header)
Jof = {8:J1, 24:J2, 32:J3, 40:J4}

# Energy per spin (drop the 1/2 double-count and 1/4 sublattice average: they
# are positive constants and do not move the minimum).
E = sum(Jof[d2] * sum(cos(q*dz) for dz in shell_dz(d2)) for d2 in Jof)
E = E.trig_reduce().simplify_full()
print("E(q)/const =", E)

# --- stationarity: minimise over q.  dE/dq = 0.
# Every shell sum is a polynomial in c = cos(2q): the interior extrema (sin(2q)!=0)
# are the roots of dE/dc, with cos4q = 2c^2-1 and cos6q = 4c^3-3c.
c = var('c')
A = J1*4 + J2*8 + J4*4          # cos(2q) coefficient of E
B = J2*4 + J3*8                 # cos(4q) coefficient
D = J4*4                        # cos(6q) coefficient  (J4 only)
Ec = A*c + B*(2*c^2-1) + D*(4*c^3-3*c)
stat = Ec.diff(c)               # dE/dc = 0 at interior extrema (c in (-1,1))
print("\nStationarity  dE/dc = 0  :", stat.expand(), "= 0")

# --- (1) generalised J3 given target Q, J2, J4  (linear in J3 -> closed form)
J3_sol = solve(stat == 0, J3)[0].rhs().simplify_full()
print("\nJ3(c, J2, J4) =", J3_sol)
Qsym = var('Q'); c_of_Q = cos(2 * Q*2*pi/8)
J3_of_Q = J3_sol.subs(c == c_of_Q).simplify_full()
print("J3(Q,J2,J4)   =", J3_of_Q)

# --- (2) extract minimum Q given all couplings (quadratic in c when J4!=0)
def Q_min(j2, j3, j4):
    poly = stat.subs(J2==j2, J3==j3, J4==j4)
    roots = [r for r in solve(poly==0, c, solution_dict=True)]
    cs = [rr[c] for rr in roots]
    best = None
    for cval in cs:
        cval = cval.n()
        if -1 < cval < 1:
            Qv = (arccos(cval)/2) / (2*pi/8)        # invert c=cos(2 q_z), q_z=Q*2pi/8
            Ev = Ec.subs(J2==j2,J3==j3,J4==j4, c==cval).n()
            if best is None or Ev < best[1]:
                best = (Qv.n(), Ev)
    return None if best is None else best[0]

# sanity: reproduce J3_from_Qz (J4=0) and show a J4-shifted example
print("\n-- checks (J1=-1) --")
for j2,Q0 in [(0.0,0.7),(0.1,0.9),(0.2,0.5)]:
    j3 = J3_of_Q.subs(J2==j2, J4==0, Q==Q0).n()
    print(f"J2={j2} Q0={Q0}: J3={float(j3):+.5f}  -> Q_min={float(Q_min(j2,j3,0.0)):.4f}")
j2,j3,j4 = 0.0, 0.27534, 0.05
print(f"With J4={j4}: same J3={j3} now gives Q_min={float(Q_min(j2,j3,j4)):.4f} (was 0.70)")
