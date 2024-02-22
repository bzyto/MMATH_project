import cv2
from scipy import ndimage
import numpy as np
from PoissonLinearApprox import Triangle, Mesh, Vertex, PoissonSolver
from mesh_gen import get_boundary, generate_vertices, generate_mytriangles
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
#Parameters for the ellipse
a = 5  # semi-major axis
b = 3  # semi-minor axis
n_points = 100  # number of points

# Generate points on the ellipse
theta = np.linspace(0, 2*np.pi, n_points)
x = a * np.cos(theta)
y = b * np.sin(theta)
points = np.vstack((x, y)).T

# Create the Delaunay triangulation
tri = Delaunay(points)
mesh = Mesh(generate_vertices(tri), generate_mytriangles(tri), 0)
solution = PoissonSolver(mesh, 0)
solution =solution.solve()

plt.figure()
plt.triplot(triangulation, '-')
plt.title('TU Delft logo triangular mesh')
plt.xticks([])  # Hide x-axis ticks
plt.yticks([]) 
plt.gca().invert_yaxis()
plt.text(0.5, -0.1, 'Mesh resolution sacrificed for readibility', ha='center', va='center', transform=plt.gca().transAxes)
plt.show()
# Plot the solution
plt.figure()
plt.tripcolor(triangulation, solution, shading='flat', cmap='gnuplot')
plt.colorbar(label = "Temperature")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r'Poisson equation $\Delta u = -1$ with homogenous boundary condition '+ '\n solved on the TU Delft logo domain')

# Invert y-axis
plt.gca().invert_yaxis()
plt.savefig("delft_interview.jpg", dpi = 300)
plt.figure()
plt.tricontourf(triangulation , solution, cmap='gnuplot')
plt.colorbar(label='Temperature')
plt.title(r'Contour plot of the solution to the Poisson problem $\Delta u =-1$,' + "\n with $u=0$ on the boundary, on the TU Delft logo")
plt.gca().invert_yaxis()

plt.show()