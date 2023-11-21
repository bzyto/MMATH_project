import PoissonLinearApprox as c0
import PoissonZ3 as z3
import numpy as np
import matplotlib.pyplot as plt
vals = np.loadtxt("comparison.txt")
triangles = np.loadtxt("triangles.txt")
# vals = []
# triangles = []
reference_val = 0.07367135
he = [1/n for n in range(2, 50, 2)]
# for h in he:
#     m1 = c0.generateMesh_UnitSquare(h)
#     triangles.append(len(m1.triangles))
#     m2 = z3.generateMesh_UnitSquare(h)
#     s1 = c0.PoissonSolver(m1, 0)
#     s2 = z3.PoissonZ3Solver(m2)
#     vals.append([abs(np.max(s1.solve())-reference_val), abs(np.max(s2.Solve()[::3])-reference_val)])
#     print("Done with h = ", h)
# np.savetxt("comparison.txt", vals)
# np.savetxt("triangles.txt", triangles)
h_inv = 1/np.array(he)
plt.plot(h_inv, vals[:,0], 'rx', label="linear")
plt.plot(h_inv, vals[:,1], 'bx', label="Z3")
plt.legend()
plt.xlabel("1/h")
plt.ylabel("log(Absolute Error)")
plt.yscale("log")
plt.grid()
plt.title("Absolute error of the finite element solution relative to the reference value \n at the centre of the domain for different mesh sizes")

plt.savefig("comparison.png", dpi = 300)

