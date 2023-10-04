import numpy as np
import matplotlib.pyplot as plt
import time
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
class Triangle:
    def __init__(self, vertices, VertexNumbers, GlobalNumber = 0):
        self.x_0 = vertices[0]
        self.x_1 = vertices[1]
        self.x_2 = vertices[2]
        self.GlobalNumber = GlobalNumber
        self.VertexNumbers = VertexNumbers
    def area(self):
        return 0.5 * np.abs(np.cross(self.x_1 - self.x_0, self.x_2 - self.x_0))
    def __str__(self):
        return "Triangle with vertices \n" + str(self.x_0) + "\n" + str(self.x_1) + "\n" + str(self.x_2) + "\n And global number " + str(self.GlobalNumber)
    def COM(self):
        x_coord = (1/3)*(self.x_0[0]+self.x_1[0]+self.x_2[0])
        y_coord = (1/3)*(self.x_0[1]+self.x_1[1]+self.x_2[1])
        return [x_coord, y_coord]
class Vertex:
    def __init__(self, coordinates, global_number):
        self.coordinates = coordinates
        self.global_number = global_number
    def __str__(self):
        return "Vertex with coordinates " +str(self.coordinates)+" and global number " +str(self.global_number)
class Mesh:
    def __init__(self, vertices, triangles):
        self.vertices = vertices
        self.triangles = triangles
    def __str__(self):
        return "Mesh with " + str(len(self.vertices)**2) + " vertices and " + str(len(self.triangles)) + " triangles"
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

def generateMesh_UnitSquare(h = 0.2):
    x = y = np.linspace(0, 1, int(1/h)+1)
    x_grid, y_grid = np.meshgrid(x, y)
    vertices = []
    triangles = []
    loopcounter = 0
    for i in range(len(x_grid)):
        for j in range(len(x_grid)):
            vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter))
            loopcounter+=1
    vertices = np.array(vertices)
    vertices = vertices.reshape(len(x), len(y))
    loopcounter = 0
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            v1 = vertices[i][j]
            v2 = vertices[i+1][j]
            v3 = vertices[i+1][j+1]
            v4 = vertices[i][j+1]
            triangles.append(Triangle([v1.coordinates, v2.coordinates, v4.coordinates], [v1.global_number, v2.global_number, v4.global_number], loopcounter))
            triangles.append(Triangle([v4.coordinates, v2.coordinates, v3.coordinates], [v4.global_number, v2.global_number, v3.global_number], loopcounter+1))
            loopcounter+=2
            
    return Mesh(vertices, triangles)
def main():
    start = time.time()
    mesh = generateMesh_UnitSquare(0.25)
    print(f"Took {time.time()- start} seconds")
    mesh.plot(numbering=True)
    print(mesh.ConnectivityMatrix())
    #print(mesh)


if __name__ == "__main__":
    main()