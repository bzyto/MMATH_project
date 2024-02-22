import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, vertices, triangles, h, nomids):
        self.vertices = vertices
        self.triangles = triangles
        self.h = h
        self.nomids = nomids
    def __str__(self):
        return "Mesh with " + str(len(self.vertices)) + " vertices and " + str(len(self.triangles)) + " triangles"
    def plot(self, numbering = False):
        for triangle in self.triangles:
            plt.plot([triangle.x_0[0], triangle.x_1[0]], [triangle.x_0[1], triangle.x_1[1]], 'k-')
            plt.plot([triangle.x_1[0], triangle.x_2[0]], [triangle.x_1[1], triangle.x_2[1]], 'k-')
            plt.plot([triangle.x_2[0], triangle.x_0[0]], [triangle.x_2[1], triangle.x_0[1]], 'k-')
            if numbering:
                # plt.text(triangle.COM()[0], triangle.COM()[1], str(fr"\textcircled{triangle.GlobalNumber}")) #color = 'red')
                plt.text(triangle.x_0[0], triangle.x_0[1], str(triangle.VertexNumbers[0:3]))
                plt.text(triangle.x_1[0], triangle.x_1[1], str(triangle.VertexNumbers[3:6]))
                plt.text(triangle.x_2[0], triangle.x_2[1], str(triangle.VertexNumbers[6:9]))
        plt.title(fr"Triangular mesh with $h$ = {self.h}")
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$', rotation = 0)
        plt.show()
    def WhatsInside(self):
        for triangle in self.triangles:
            print(triangle)
    def ConnectivityMatrix(self):
        CMatrix = np.zeros((len(self.triangles), 12))
        for i in range(len(self.triangles)):
            for j in range(12):
                CMatrix[i][j] = self.triangles[i].VertexNumbers[j]
        return CMatrix
    def MapSolution(self, solution):
        for index, value in enumerate(solution.flatten()):
            try:
                self.vertices[index].val = value
            except IndexError:
                return