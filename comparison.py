
# import numpy as np
import matplotlib.pyplot as plt
# z3 = np.loadtxt('text_files/sinfucntion_comp_z3.txt')
# p1 = np.loadtxt('text_files/sinfunction_comp_p1.txt')
plt.rcParams['text.usetex'] = True #for nice plots
# plt.plot(z3[0:,],z3[:,1], 'r-x', label="$Z3$")
# plt.plot(p1[0:,],p1[:,1], 'b-x', label="$P1$")
# plt.xlabel("$\log h$")
# plt.ylabel("$\log$(Absolute Error)")
# plt.show()
x = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
y = [0.005294835076636417, 0.0011218500699520517, 0.00026615103311606464, 6.563214912822142e-05, 2.7202708512252816e-05]
x1 = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
y1 = [0.049846516630434046, 0.011781677515873445, 0.003206575608387441, 0.0007989377823770516, 0.00020077342543534105, 5.0182773209783704e-05, 1.25497562291077e-05]

# Create the plot
plt.figure()
plt.plot(x1, y1, 'bx-', label = "$P1$")
plt.plot(x, y, 'rx-', label = "$Z3$")
plt.xlabel("$\log h$")
plt.ylabel("$\log$(Absolute Error)")
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Maximum absolute error at a vertex in the triangular mesh')
plt.savefig('figures/sincomp.png', dpi = 300)