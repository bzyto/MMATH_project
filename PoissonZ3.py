import numpy as np
import matplotlib.pyplot as plt
class Triangle:
    def __init__(self, vertices, VertexNumbers, GlobalNumber = 0):
        self.x_0 = np.array(vertices[0])
        self.x_1 = np.array(vertices[1])
        self.x_2 = np.array(vertices[2])
        self.vertices = vertices
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
    def LocalCubic(self):
        # we need to compute the coefficients of all 10 local approximation functions
        # a local approximation function phi_i(x,y) is given by c_0 + c_1*x+c_2*y+c_3*x^2+ c_4 y^2 + c_5*x*y + c_6 x^3 + c_7 y^3+c_8 x^2y + c_9 xy^2.
        # we will compute the coefficient 10 times, for 10 local approximation functions
        # lhs will be the same for all of the systems
        # we construct the lhs row by row
        row_1 = np.array([1, self.x_0[0], self.x_0[1], self.x_0[0]**2, self.x_0[1]**2, self.x_0[0]*self.x_0[1], self.x_0[0]**3, self.x_0[1]**3, self.x_0[0]**2*self.x_0[1], self.x_0[0]*self.x_0[1]**2]) ##phi(x_0, y_0)
        row_2 = np.array([0, 1, 0, 2*self.x_0[0], 0, self.x_0[1], 3*self.x_0[0]**2, 0, 2*self.x_0[0]*self.x_0[1], self.x_0[1]**2]) ## d/dx phi(x_0, y_0)
        row_3 = np.array([0, 0, 1, 0, 2*self.x_0[1], self.x_0[0], 0, 3*self.x_0[1]**2, self.x_0[0]**2, 2*self.x_0[0]*self.x_0[1]]) ## d/dy phi(x_0, y_0)
        row_4 = np.array([1, self.x_1[0], self.x_1[1], self.x_1[0]**2, self.x_1[1]**2, self.x_1[0]*self.x_1[1], self.x_1[0]**3, self.x_1[1]**3, self.x_1[0]**2*self.x_1[1], self.x_1[0]*self.x_1[1]**2]) ##phi(x_1, y_1)
        row_5 = np.array([0, 1, 0, 2*self.x_1[0], 0, self.x_1[1], 3*self.x_1[0]**2, 0, 2*self.x_1[0]*self.x_1[1], self.x_1[1]**2])
        row_6 = np.array([0, 0, 1, 0, 2*self.x_1[1], self.x_1[0], 0, 3*self.x_1[1]**2, self.x_1[0]**2, 2*self.x_1[0]*self.x_1[1]]) 
        row_7 = np.array([1, self.x_2[0], self.x_2[1], self.x_2[0]**2, self.x_2[1]**2, self.x_2[0]*self.x_2[1], self.x_2[0]**3, self.x_2[1]**3, self.x_2[0]**2*self.x_2[1], self.x_2[0]*self.x_2[1]**2])
        row_8 = np.array([0, 1, 0, 2*self.x_2[0], 0, self.x_2[1], 3*self.x_2[0]**2, 0, 2*self.x_2[0]*self.x_2[1], self.x_2[1]**2])
        row_9 = np.array([0, 0, 1, 0, 2*self.x_2[1], self.x_2[0], 0, 3*self.x_2[1]**2, self.x_2[0]**2, 2*self.x_2[0]*self.x_2[1]])
        row_10 = np.array([1, self.COM()[0], self.COM()[1], self.COM()[0]**2, self.COM()[1]**2, self.COM()[0]*self.COM()[1], self.COM()[0]**3, self.COM()[1]**3, self.COM()[0]**2*self.COM()[1], self.COM()[0]*self.COM()[1]**2]) #phi at com 
        LHS = np.array([row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10])
        solution = []
        for i in range(10):
            RHS = np.zeros(10)
            RHS[i] = 1
            solution.append(np.linalg.solve(LHS, RHS))
        return np.array(solution)


class Vertex:
    def __init__(self, coordinates, global_number, boundary):
        self.coordinates = coordinates
        self.global_number = global_number
        self.boundary = boundary
    def __str__(self):
        return "Vertex with coordinates " +str(self.coordinates)+" and global number " +str(self.global_number)
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
                # plt.text(triangle.x_0[0], triangle.x_0[1], str(triangle.VertexNumbers[0]))
                # plt.text(triangle.x_1[0], triangle.x_1[1], str(triangle.VertexNumbers[1]))
                # plt.text(triangle.x_2[0], triangle.x_2[1], str(triangle.VertexNumbers[2]))
        plt.title(f"Triangular mesh with h = {self.h}")
        plt.savefig("mesh.jpg")
    def WhatsInside(self):
        for triangle in self.triangles:
            print(triangle)
    def ConnectivityMatrix(self):
        CMatrix = np.zeros((len(self.triangles), 3))
        for i in range(len(self.triangles)):
            for j in range(3):
                CMatrix[i][j] = self.triangles[i].VertexNumbers[j]
        return CMatrix
class PoissonZ3Solver:
    def __init__(self, mesh, bc):
        self.mesh = mesh
        self.bc = bc
    def IntegrationMatrixLHS(self):
        n_triangles = len(self.mesh.triangles)
        ElementMatrix = np.zeros((n_triangles, 10, 10))
        
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
def main():
    mesh = generateMesh_UnitSquare(1/2)
    mesh.plot(numbering=True)
main()
