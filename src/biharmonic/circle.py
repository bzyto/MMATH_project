import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True #for nice plots
r = 1
def generate_circle_points(intervals, r=1):
    circ = []
    for i in range(intervals+1):
        circ.append([r*np.cos(i*np.pi/(2*intervals)), r*np.sin(i*np.pi/(2*intervals))])
    return circ
circ = generate_circle_points(100)
c_1 = generate_circle_points(1)
c_2 = generate_circle_points(2)
c_3 = generate_circle_points(3)
c_5 = generate_circle_points(4)
fig, axs = plt.subplots(2, 2)

# Plot c_1 and the exact circle on the first subplot
axs[0, 0].plot(*zip(*c_1), label='Linearized', linestyle='--', color='blue', marker = 'x')
axs[0, 0].plot(*zip(*circ), label='Exact', color='red')
axs[0, 0].set_xlabel('$x$')
axs[0, 0].set_ylabel('$y$')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Plot c_2 and the exact circle on the second subplot
axs[0, 1].plot(*zip(*c_2), label='Linearized', linestyle='--', color='blue', marker = 'x')
axs[0, 1].plot(*zip(*circ), label='Exact', color='red')
axs[0, 1].set_xlabel('$x$')
axs[0, 1].set_ylabel('$y$')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Plot c_3 and the exact circle on the third subplot
axs[1, 0].plot(*zip(*c_3), label='Linearized', linestyle='--', color='blue', marker = 'x')
axs[1, 0].plot(*zip(*circ), label='Exact', color='red')
axs[1, 0].set_xlabel('$x$')
axs[1, 0].set_ylabel('$y$')
axs[1, 0].grid(True)
axs[1, 0].legend()

# Plot c_5 and the exact circle on the fourth subplot
axs[1, 1].plot(*zip(*c_5), label='Linearized', linestyle='--', color='blue', marker = 'x')
axs[1, 1].plot(*zip(*circ), label='Exact', color='red')
axs[1, 1].set_xlabel('$x$')
axs[1, 1].set_ylabel('$y$')
axs[1, 1].grid(True)
axs[1, 1].legend()
fig.suptitle("Circular boundary and its linearizations")
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig("circle.jpg", dpi = 5
             myvals = [0.009306, 0.018383, 0.020049, 0.020229, 0.020244, 0.020245]
# ref = 0.020245
# myers = []
# DavidVals = [0.022716,0.020803, 0.020333, 0.020257, 0.020247, 0.020245]
# daviderors =[]
# for i in range(6):
#     daviderors.append(np.abs(ref-DavidVals[i]))
#     myers.append(ref-myvals[i])
# print(daviderors)
# plt.figure()
# plt.loglog([1/2, 1/4, 1/8, 1/16, 1/32, 1/64], myers, marker='x', label = '$CT$')
# plt.loglog([1/2, 1/4, 1/8, 1/16, 1/32, 1/64], daviderors, marker='x', label = 'quadrilateral')
# plt.loglog([1/2, 1/4, 1/8, 1/16, 1/32, 1/64], [1/4, 1/16, 1/64, 1/256, 1/1024, 1/4096], linestyle='--', label = '$h^2$')
# plt.loglog([1/2, 1/4, 1/8, 1/16, 1/32, 1/64], [1/8, 1/64, 1/512, 1/4096, 1/32768, 1/262144], linestyle='--', label = '$h^3$')
# plt.ylabel('$\log$(Absolute Error)')
# plt.xlabel('$\log h$')
# plt.xticks([10**-2, 10**-1, 10**0], ['$10^{-2}$', "$10^{-1}$", '$10^{0}$'])
# plt.grid()
# plt.legend()
# plt.title('Absolute error of the finite element solution relative to the reference value \n at the centre of the domain')
# plt.savefig("biharm_error.jpg", dpi = 400)00)