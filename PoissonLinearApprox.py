import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import matplotlib.tri as mtri
from scipy import integrate, sparse
plt.rcParams['text.usetex'] = True #for nice plots

"""
Code to solve the poisson equation using Galerkin Finite Element Method on a triangular mesh
We take the domain to be the unit square 
The equation to be solved is
-Δu = 1 inside the domain
u = 0 on the boundary
"""
### TODO
### Mesh generation
### Galerkin Matrix 
### ASSEMBly
### solve the linear system 
### Plot results
def gaussian_quad(func, triangle):
    # Quadrature weights and points in barycentric coordinates
    weights = np.array([0.225000000000000, 0.125939180544827, 0.125939180544827, 0.125939180544827, 0.132394152788506, 0.132394152788506, 0.132394152788506])
    barycentric_coordinates = np.array([[0.333333333333333, 0.333333333333333], [0.797426985353087, 0.101286507323456], [0.101286507323456, 0.797426985353087], [0.101286507323456, 0.101286507323456], [0.059715871789770, 0.470142064105115], [0.470142064105115, 0.059715871789770], [0.470142064105115, 0.470142064105115]])

    # Vertices of the triangle
    A = triangle.x_0
    B = triangle.x_1
    C = triangle.x_2

    # Area of the triangle

    # Initialize the result
    result = 0

    # Loop over the quadrature points
    for i in range(len(weights)):
        # Convert the barycentric coordinates to Cartesian coordinates
        alpha, beta = barycentric_coordinates[i]
        gamma = 1 - alpha - beta
        P = alpha * A + beta * B + gamma * C

        # Evaluate the function at the quadrature point and add to the result
        result += weights[i] * func(*P)

    # Multiply by the area of the triangle
    result *= triangle.area()

    return result
def gaussian_quad_v(func, triangle):
    # Quadrature weights and points in barycentric coordinates
    weights = np.array([0.225000000000000, 0.125939180544827, 0.125939180544827, 0.125939180544827, 0.132394152788506, 0.132394152788506, 0.132394152788506])
    barycentric_coordinates = np.array([[0.333333333333333, 0.333333333333333], [0.797426985353087, 0.101286507323456], [0.101286507323456, 0.797426985353087], [0.101286507323456, 0.101286507323456], [0.059715871789770, 0.470142064105115], [0.470142064105115, 0.059715871789770], [0.470142064105115, 0.470142064105115]])

    # Vertices of the triangle
    A = triangle.x_0
    B = triangle.x_1
    C = triangle.x_2

    # Convert the barycentric coordinates to Cartesian coordinates
    gamma = 1 - np.sum(barycentric_coordinates, axis=1)
    P = np.outer(barycentric_coordinates[:, 0], A) + np.outer(barycentric_coordinates[:, 1], B) + np.outer(gamma, C)

    # Evaluate the function at the quadrature points
    func_values = np.apply_along_axis(*func, 1, P)

    # Calculate the integral
    result = triangle.area() * np.dot(weights, func_values)

    return result
class Triangle:
    def __init__(self, vertices, VertexNumbers, GlobalNumber = 0):
        self.x_0 = np.array(vertices[0])
        self.x_1 = np.array(vertices[1])
        self.x_2 = np.array(vertices[2])
        self.GlobalNumber = GlobalNumber
        self.VertexNumbers = VertexNumbers
        self.bvector = [self.x_1[1]- self.x_2[1], self.x_2[1]- self.x_0[1], self.x_0[1]- self.x_1[1]] #useful later
        self.cvector = [self.x_2[0]- self.x_1[0], self.x_0[0]- self.x_2[0], self.x_1[0]- self.x_0[0]]
    def area(self):
        return 0.5 * np.abs(np.cross(self.x_1 - self.x_0, self.x_2 - self.x_0))
    def __str__(self):
        return "Triangle with vertices \n" + str(self.x_0) + "\n" + str(self.x_1) + "\n" + str(self.x_2) + "\n And global number " + str(self.GlobalNumber)
    def COM(self):
        ##centre of mass
        x_coord = (1/3)*(self.x_0[0]+self.x_1[0]+self.x_2[0])
        y_coord = (1/3)*(self.x_0[1]+self.x_1[1]+self.x_2[1])
        return [x_coord, y_coord]
    def LocalLinear(self):
        row_1 = np.array([1, self.x_0[0], self.x_0[1]])
        row_2 = np.array([1, self.x_1[0], self.x_1[1]])
        row_3 = np.array([1, self.x_2[0], self.x_2[1]])
        LHS = np.array([row_1, row_2, row_3])
        soln = []
        for i in range(3):
            RHS = np.zeros(3)
            RHS[i] = 1
            soln.append(np.linalg.solve(LHS, RHS))
        return np.array(soln)

class Vertex:
    def __init__(self, coordinates, global_number, boundary, val = None):
        self.coordinates = coordinates
        self.global_number = global_number
        self.boundary = boundary
        self.val = val
    def __str__(self):
        return "Vertex with coordinates " +str(self.coordinates)+" and global number " +str(self.global_number)+ " and value"+str(self.val)
class Mesh:
    def __init__(self, vertices, triangles, h):
        self.vertices = vertices
        self.triangles = triangles
        self.h = h
    def __str__(self):
        return "Mesh with " + str(len(self.vertices)) + " vertices and " + str(len(self.triangles)) + " triangles"
    def plot(self, numbering = False):
        for triangle in self.triangles:
            plt.plot([triangle.x_0[0], triangle.x_1[0]], [triangle.x_0[1], triangle.x_1[1]], 'k-')
            plt.plot([triangle.x_1[0], triangle.x_2[0]], [triangle.x_1[1], triangle.x_2[1]], 'k-')
            plt.plot([triangle.x_2[0], triangle.x_0[0]], [triangle.x_2[1], triangle.x_0[1]], 'k-')
            if numbering:
                plt.text(triangle.COM()[0], triangle.COM()[1], str(triangle.GlobalNumber), color = 'red')
                plt.text(triangle.x_0[0], triangle.x_0[1], str(triangle.VertexNumbers[0]))
                plt.text(triangle.x_1[0], triangle.x_1[1], str(triangle.VertexNumbers[1]))
                plt.text(triangle.x_2[0], triangle.x_2[1], str(triangle.VertexNumbers[2]))
        plt.title(f"Triangular mesh with $h = {self.h}$")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()
    def WhatsInside(self):
        for triangle in self.triangles:
            print(triangle)
    def ConnectivityMatrix(self):
        CMatrix = np.zeros((len(self.triangles), 3))
        for i in range(len(self.triangles)):
            for j in range(3):
                CMatrix[i][j] = self.triangles[i].VertexNumbers[j]
        return CMatrix
    def MapSolution(self, solution):
        for index, value in enumerate(solution.flatten()):
            self.vertices[index].val = value
class PoissonSolver:
    def __init__(self, mesh, bc):
        self.mesh = mesh
        self.bc = bc
    def ElementIntegrationLHS(self):
        n_triangles = len(self.mesh.triangles)
        IntegrationMatrix = np.zeros((n_triangles, 3, 3))
        for k in range(n_triangles):
            c = self.mesh.triangles[k].LocalLinear()
            i, j = np.indices((3, 3))
            mask = j>=i
            v_0 = np.outer(c[:, 1], c[:, 1]) + np.outer(c[:, 2], c[:, 2])   
            f = lambda x, y: v_0
            IntegrationMatrix[k] = np.where(mask, gaussian_quad(f, self.mesh.triangles[k]), IntegrationMatrix[k].T)
            IntegrationMatrix[k] = np.where(mask, IntegrationMatrix[k], IntegrationMatrix[k].T)
        return IntegrationMatrix
    def ElementIntegrationRHS(self):
        n_triangles = len(self.mesh.triangles)
        WRHSVector = np.zeros((3, n_triangles))
        for k in range(n_triangles):
            c = self.mesh.triangles[k].LocalLinear()
            ### non steady f
            f = lambda x, y: (c[:, 0] + c[:, 1]*x + c[:, 2]*y)*2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
            WRHSVector[:, k] = gaussian_quad(f, self.mesh.triangles[k])
        return WRHSVector
    def Assembly(self):
        ElementMatrix = self.ElementIntegrationLHS()
        ElementRHS = self.ElementIntegrationRHS()
        ConnectivityMatrix = self.mesh.ConnectivityMatrix()
        matrixSize = int(len(self.mesh.vertices))
        LHS = np.zeros((matrixSize, matrixSize))
        RHS = np.zeros(matrixSize)
        ## loop as in https://eprints.maths.manchester.ac.uk/894/2/0-19-852868-X.pdf p.26
        for k in range(len(ConnectivityMatrix)):
            for j in range(3):
                for i in range(3):
                    LHS[int(ConnectivityMatrix[k][i])][int(ConnectivityMatrix[k][j])]+=ElementMatrix[k][i][j]
                RHS[int(ConnectivityMatrix[k][j])]+=ElementRHS[j][k]
        ## at this point, the galerkin matrix is of size n by n where n is the total number of vertices
        ## we have enforce the boundary conditions on the system
        for i in range(matrixSize):
            if self.mesh.vertices[i].boundary:
                LHS[i, :] = 0
                LHS[i][i] = 1
                RHS[i] = 0
        return LHS, RHS
    def Assembly_without_bc(self):
        ElementMatrix = self.ElementIntegrationLHS()
        ElementRHS = self.ElementIntegrationRHS()
        ConnectivityMatrix = self.mesh.ConnectivityMatrix()
        matrixSize = int(len(self.mesh.vertices))
        LHS = np.zeros((matrixSize, matrixSize))
        RHS = np.zeros(matrixSize)
        ## loop as in https://eprints.maths.manchester.ac.uk/894/2/0-19-852868-X.pdf p.26
        for k in range(len(ConnectivityMatrix)):
            for j in range(3):
                for i in range(3):
                    LHS[int(ConnectivityMatrix[k][i])][int(ConnectivityMatrix[k][j])]+=ElementMatrix[k][i][j]
                RHS[int(ConnectivityMatrix[k][j])]+=ElementRHS[j][k]
        return LHS, RHS
    def solve(self):
        LHS, RHS = self.Assembly()
        #matrix_to_binary_image(LHS)
        start = time.time()
        solution = sparse.linalg.spsolve(sparse.csr_matrix(LHS), RHS)##taking advantage of the sparsity of the matrix
        return solution
    def plot(self):
        soln = self.solve()
        n = len(soln)
        soln = to_matrix(soln)
        ## inspired by https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html#sphx-glr-gallery-mplot3d-trisurf3d-2-py
        x = y = np.linspace(0, 1, int(1/self.mesh.h) +1 )
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        z = soln.flatten()
        tri = mtri.Triangulation(x, y)
        fig = plt.figure()
        ax = fig.add_subplot( projection='3d')
        ax.scatter(x, y, z)
        plt.show()
    def plot_contour(self):
        soln = self.solve()
        n = len(soln)
        soln = soln.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
        ## inspired by https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html#sphx-glr-gallery-mplot3d-trisurf3d-2-py
        x = y = np.linspace(0, 1, int(1/self.mesh.h) +1 )
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        z = soln.flatten()
        tri = mtri.Triangulation(x, y)
        plt.figure()
        plt.tricontourf(tri , z, cmap='gnuplot')
        # plt.colorbar(label='Solution')
        plt.title(r'Contour plot of the solution to the Poisson problem $\Delta u =-1$,' + "\n with $u=0$ on the boundary, on the unit square")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()
    def plot_triangle(self):
        soln = self.solve()
        n = len(soln)
        soln = soln.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
        ## inspired by https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html#sphx-glr-gallery-mplot3d-trisurf3d-2-py
        x = y = np.linspace(0, 1, int(1/self.mesh.h) +1 )
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        z = soln.flatten()
        tri = mtri.Triangulation(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral_r )
        ax.set(
            xlabel = 'x',
            ylabel = 'y',
            zlabel = 'z'
        )
        ax.set_title(f'Poisson equation solution with $h = {self.mesh.h}$')
        plt.show()
    def plot_triangle_save(self):
        soln = self.solve()
        n = len(soln)
        soln = soln.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
        ## inspired by https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html#sphx-glr-gallery-mplot3d-trisurf3d-2-py
        x = y = np.linspace(0, 1, int(1/self.mesh.h) +1 )
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        z = soln.flatten()
        tri = mtri.Triangulation(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral_r )
        ax.set(
            xlabel = '$x$',
            ylabel = '$y$',
            zlabel = '$z$'
        )
        ax.set_title(f'Poisson equation solution with $h = {self.mesh.h}$')
        plt.savefig(fr"figures/P0Poisson_equation_h_{self.mesh.h}.jpg", dpi=300)



def generateMesh_UnitSquare(h = 0.2):
    x = y = np.linspace(0, 1, int(1/h)+1)
    x_grid, y_grid = np.meshgrid(x, y)
    vertices = []
    triangles = []
    loopcounter = 0
    for i in range(len(x_grid)):
        for j in range(len(x_grid)):
            boundary = False
            if x_grid[i][j] == 0 or x_grid[i][j] == 1 or y_grid[i][j] == 0 or y_grid[i][j] == 1:
                boundary = True
            vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter, boundary))
            loopcounter+=1
    vertices = np.array(vertices)
    vertices = vertices.reshape(len(x), len(y))
    loopcounter = 0
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            ## we take care both of the "real" coordinates of the vertices and their numbering
            ## as a result we are able to create both mesh and the related connectivity matrix
            ## change the orientation of the triangles
            v1 = vertices[i][j]
            v2 = vertices[i+1][j]
            v3 = vertices[i+1][j+1]
            v4 = vertices[i][j+1]
            triangles.append(Triangle([v1.coordinates, v2.coordinates, v3.coordinates], [v1.global_number, v2.global_number, v3.global_number], loopcounter)) ##lower triangle
            triangles.append(Triangle([v4.coordinates, v1.coordinates, v3.coordinates], [v4.global_number, v1.global_number, v3.global_number], loopcounter+1)) ## upper triangle
            loopcounter+=2
    vertices = vertices.reshape((len(x)*len(y)))
    return Mesh(vertices, triangles, h)
def generateFunnyMesh(h):
    x = y = np.linspace(0, 1, int(1/h)+1)
    x_grid, y_grid = np.meshgrid(x, y)
    vertices = []
    triangles = []
    loopcounter = 0
    for i in range(len(x_grid)):
        for j in range(len(x_grid)):
            boundary = False
            if x_grid[i][j] == 0 or x_grid[i][j] == 1 or y_grid[i][j] == 0 or y_grid[i][j] == 1:
                boundary = True
            if not (x_grid[i][j]<=0.5 and y_grid[i][j]>=0.5):
                vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter, boundary))
                loopcounter+=1
    vertices = np.array(vertices)
    vertices = vertices.reshape(len(x), len(y))
    loopcounter = 0
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            ## we take care both of the "real" coordinates of the vertices and their numbering
            ## as a result we are able to create both mesh and the related connectivity matrix
            ## change the orientation of the triangles
            v1 = vertices[i][j]
            v2 = vertices[i+1][j]
            v3 = vertices[i+1][j+1]
            v4 = vertices[i][j+1]
            triangles.append(Triangle([v1.coordinates, v2.coordinates, v3.coordinates], [v1.global_number, v2.global_number, v3.global_number], loopcounter)) ##lower triangle
            triangles.append(Triangle([v4.coordinates, v1.coordinates, v3.coordinates], [v4.global_number, v1.global_number, v3.global_number], loopcounter+1)) ## upper triangle
            loopcounter+=2
    vertices = vertices.reshape((len(x)*len(y)))
    return Mesh(vertices, triangles, h)
def to_matrix(arr):
    # calculate the size of the output matrix
    n = int(np.sqrt(len(arr)))
    arr = np.array(arr)
    arr = arr.reshape(n, n)
    n = n+2
    # create a matrix of zeros with the appropriate shape
    mat = np.zeros((n, n))
    
    # fill in the non-zero elements
    for i in range(n-2):
        for j in range(n-2):
            mat[i+1][j+1] = arr[i][j]
    return mat
def matrix_to_binary_image(matrix):
    # Create a binary matrix where 1 represents nonzero entries
    binary_matrix = np.where(matrix != 0, 1, 0)

    # Plot the binary image
    plt.imshow(binary_matrix, cmap='binary', interpolation='nearest')
    plt.show()
def u_exact(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)
def main():
    for h in [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]:
        mesh = generateMesh_UnitSquare(h)
        solution = PoissonSolver(mesh, 0)
        a = solution.solve()
        mesh.MapSolution(a)
        error_arr = np.array([])
        for vertex in mesh.vertices[::3]:
            error_arr= np.append(error_arr, abs(vertex.val-u_exact(vertex.coordinates[0],vertex.coordinates[1])))
        print(h, np.max(error_arr))
if __name__ == "__main__":
    main()