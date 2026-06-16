import numpy as np
import matplotlib.pyplot as plt

J3 = np.load("J3_arr.npy")
k_arr = np.load("min_q.npy")

fix, ax = plt.subplots()

ax.plot(J3, k_arr)
ax.set_xlabel('J3')
ax.set_ylabel(r'$Q_{\rm min}$')

Qz = np.linspace(0,np.pi,100)
J2=-0.15

J3_pred = 1/2 * ( ( np.sin( Qz ) + ( 3 * np.sin( 3 * Qz ) + 5 * \
np.sin( 5 * Qz ) ) ) )**( -1 ) * ( -2 * ( -1 + J2 ) * np.sin( \
Qz ) + ( ( 3 + -9 * J2 ) * np.sin( 3 * Qz ) + -5 * J2 * np.sin( \
5 * Qz ) ) )

ax.plot(J3_pred, Qz) 

Qz_anothr =  np.atan2(np.sqrt((3 + 6*J2 + 24*J3 + np.sqrt(9 + 76*np.pow(J2,2) + 8*J2*(2 + 31*J3) + 8*J3*(13 + 42*J3)))/(J2 + 2*J3)),
   -np.sqrt((-3 + 34*J2 + 56*J3 - np.sqrt(9 + 76*np.pow(J2,2) + 8*J2*(2 + 31*J3) + 8*J3*(13 + 42*J3)))/(J2 + 2*J3)))
ax.plot(J3, Qz_anothr)

ax.set_xlim(0,np.max(J3)*1.1)
ax.set_ylim(-np.pi, np.pi)


plt.show()


