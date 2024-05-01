import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from Elements import *
from solver import *
from functions import *
import cProfile
import time
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True #for nice plots
discretizations = [1/2, 1/4, 1/8, 1/16, 1/32,1/64]
# my = []
# err =[]
# for h in discretizations:
#     mesh = generateMesh11square(h)
#     s = BiharmHCT(mesh)
#     print(h)
#     eigs, _ = s.SolveEigen(80.2)
#     print(eigs[0])
#     my.append(eigs[0])
# for val in my:
#     err.append( 80.9334 - val)
# plt.plot(discretizations, err)
# plt.show()
# myvals = [1, 89.2101, 81.8023, 81.0082, 80.9388, 80.9337]
# davidvals = [10.8743, 43.2815, 80.5516, 80.8792, 80.9262, 80.9333]
# ref = 80.9334
# for i in range(6):
#     myvals[i]-=ref
#     davidvals[i]-=ref
# plt.plot(discretizations, np.abs(myvals), label='CT', marker = 'x')
# plt.plot(discretizations, np.abs(davidvals), label='Quadrilateral', marker = 'x')
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(discretizations, np.power(discretizations, 3), label='$h^3$', linestyle = '--')
# plt.xlabel('$\log h$')
# plt.ylabel('$\log$(Absolute error)')
# plt.xticks([0.01, 0.1, 1], [r'$10^-2$', r'$10^-1$', r'$10^0$'])
# plt.grid()
# plt.title("Numerical eigenvalue $\lambda$ error vs mesh resolution")
# plt.legend()
# plt.savefig("eigenvalue_error.jpg", dpi = 400)
# mesh = generateMesh11square(1/64)
# s = BiharmHCT(mesh)
# s.Plot4eigenfunctions()
