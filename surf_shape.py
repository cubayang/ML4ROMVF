from __future__ import division  #from python 3.0, / means floating division and // means floor division, while in python 2.7, / means floor division
from __future__ import unicode_literals
import numpy as np
import scipy as sp
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def Medical_surface(L, T, xi_01, xi_02, xi_B, y, z):
    """
    Medical surface geometry
    """
    xi_0 = (1.0 - y/L)*(xi_02+(xi_01-xi_02-4.0*xi_B*z/T)*(1.0-z/T))
    return xi_0

def Modal_displacement(L, T, xi_B, y, z, t):
    """
    Modal displacement
    """
    xi_m = 0.1   #modal displacement amplitude
    m = 1        #number of half-wavelengths in the anterior-posterior direction
    n = 1        #approximate number of half-wavelengths in the inferior-superior direction
    f = 125        #vibration frequency
    omega = 2.0*np.pi*f        #angular frequency
    c = T*np.pi*f              #mucosal wave speed
    z_m = T*(0.6-0.02*xi_B)      #inflection point
    xi = xi_m*np.sin(m*np.pi*y/L)*(np.sin(omega*t)-n*(omega/c)*(z-z_m)*np.cos(omega*t))
    return xi


if __name__ == '__main__' :
    print ("Start")

    L=1.5     #length
    T=0.8     #thickness
    nL=151
    nT=81
    y = np.linspace(0, L, nL)
    z = np.linspace(0, T, nT)
    xx = np.zeros((nL, nT))
    yy = np.zeros((nL, nT))
    zz = np.zeros((nL, nT))
    yv, zv = np.meshgrid (y, z, sparse=False)
    dictShape = {
        "convergent" : [0.18, 0.03, 0.005],
        "divergent" : [0.03, 0.18, 0.005],
        "uniform" : [0.1, 0.1, 0.005]
    }
    xi_01 = dictShape["convergent"][0]
    xi_02 = dictShape["convergent"][1]
    xi_B  = dictShape["convergent"][2]
    gMedical = Medical_surface(L, T, xi_01, xi_02, xi_B, yv, zv)
#    niter = 16     #number of time iteration
    niter = 1     #number of time iteration
    nn = 0
    while nn<niter:
        t_norm = float(nn/niter)
        gModal = Modal_displacement(L, T, xi_B, yv, zv, t_norm)
        g = gMedical + gModal
        nn = nn + 1
        ##surface plot
        fig = plt.figure()
        fig.suptitle(r'Normal modes of vibration for convergent shape at $t_{norm}$=%1.4f' %t_norm)
        ax = fig.gca(projection='3d')
#        ax = fig.add_subplot(1, 3, 1, projection='3d')
        #plot surface
#        surf = ax.plot_surface(y, z, g, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.plot_wireframe(yv, zv, g, rstride=5, cstride=5)
        ax.set_xlabel(r'Y')
        ax.set_ylabel(r'Z')
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(r'\textbf{g}', fontsize=16, rotation=0)
        start, end = ax.set_xlim(0, L+0.01)
        ax.set_xticks(np.arange(start, end, 0.3))
        start, end = ax.set_ylim(0, T+0.01)
        ax.set_yticks(np.arange(start, end, 0.2))
#        ax.set_zlabel(r'$\xi$')
        ax.view_init(120, 90)
        plt.show()
        file = open(str(t_norm*10000) + ".dat", "w")
        count = 0
        for j in range(0, nT):
            for i in range(0, nL):
                count = count + 1
                xx[i,j] = g[j,i]
                yy[i,j] = y[i]
                zz[i,j] = z[j]
                file.write("%i %5.6f %5.6f %5.6f\n" % (count, xx[i,j], yy[i,j], zz[i,j]))
        file.close()

    print ("Done")
