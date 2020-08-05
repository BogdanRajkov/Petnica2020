import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos


L = 4
N = L*L
kx = ky = np.arange(0, 2*pi, step=(2*pi)/L)  # kx=ky=(0, pi/2, pi, 3pi/2)
x = y = np.arange(0, 5, step=0.05)
cx = cy = np.arange(0, 4)
cxx, cyy = np.meshgrid(cx, cy)
xx, yy = np.meshgrid(x, y)
# r = np.sqrt(xx**2 + yy**2)

psi = np.zeros((len(x), len(y)))

for ix in kx:
    for iy in ky:
        for i in range(len(x)):
            for j in range(len(y)):
                psi[i][j] = 1/np.sqrt(N) * \
                    cos(np.dot([ix, iy], [xx[i][j], yy[i][j]]))

        plt.contourf(x, y, psi)
        plt.title('k = [' + str(ix) + ', ' + str(iy) + ']')
        plt.colorbar().ax.set_ylabel('talasna funkcija')
        plt.xlim(0, 4)
        plt.ylim(0, 4)
        plt.ylabel('y')
        plt.xlabel('x')

        plt.scatter(cxx.flatten(), cyy.flatten(), c='k')

        plt.show()
