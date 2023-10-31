import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import cProfile
plt.rcParams['text.usetex'] = True #for nice plots
import time
class Triangle:
    def __init__(self, vertices, VertexNumbers, GlobalNumber = 0):
        self.x_0 = np.array(vertices[0])
        self.x_1 = np.array(vertices[1])
        self.x_2 = np.array(vertices[2])
        self.vertices = vertices
        self.GlobalNumber = GlobalNumber
        self.VertexNumbers = rearange_vertices(VertexNumbers)
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
def rearange_vertices(lst):
    ## for vertex ordering
    res = list()
    for i in lst:
        res.append(3*i)
        res.append(3*i+1)
        res.append(3*i+2)
    return res

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
                plt.text(triangle.COM()[0], triangle.COM()[1], str(fr"\textcircled{triangle.GlobalNumber}")) #color = 'red')
                plt.text(triangle.x_0[0], triangle.x_0[1], 0)#str(triangle.VertexNumbers[0]))
                plt.text(triangle.x_1[0], triangle.x_1[1], 1)#str(triangle.VertexNumbers[1]))
                plt.text(triangle.x_2[0], triangle.x_2[1], 2)#str(triangle.VertexNumbers[2]))
        plt.title(fr"Triangular mesh with $h$ = {self.h}")
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$', rotation = 0)
        plt.show()
    def WhatsInside(self):
        for triangle in self.triangles:
            print(triangle)
    def ConnectivityMatrix(self):
        CMatrix = np.zeros((len(self.triangles), 10))
        for i in range(len(self.triangles)):
            for j in range(10):
                CMatrix[i][j] = self.triangles[i].VertexNumbers[j]
        return CMatrix
class PoissonZ3Solver:
    def __init__(self, mesh, bc=0):
        self.mesh = mesh
        self.bc = bc
    def IntegrationMatrixLHS(self):
        # x = sp.Symbol('x')
        # y = sp.Symbol("y")
        # symb_vec =np.array([1, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])
        power_vec =[[0,0], [1, 0], [0, 1], [2, 0], [0,2], [1, 1], [3, 0], [0,3], [2, 1], [1,2]]
        n_triangles = len(self.mesh.triangles)
        ElementMatrix = np.zeros((n_triangles, 10, 10))
        for k in range(n_triangles):
            c = self.mesh.triangles[k].LocalCubic() #coefficient matrix
            for i in range(10):
                for j in range(10):
                    ## create a vector with c^i_kc^j_l coefficients as outlined in project
                    if j>=i:#taking advantage of the symmetry
                        v_0 = c[i][1]*c[j][1] + c[i][2]*c[j][2]
                        v_1 = 2*c[i][3]*c[j][1] + 2*c[i][1]*c[j][3] +c[i][5]*c[j][2] +c[i][2]*c[j][5]
                        v_2 = 2*c[i][4]*c[j][2] +2*c[i][4]*c[j][2] +c[i][5]*c[j][1] +c[i][1]*c[j][5]
                        v_3 = 3*c[i][6]*c[j][1] +3*c[i][1]*c[j][6] +4*c[i][3]*c[j][3] +c[i][8]*c[j][2] +c[i][2]*c[j][8] +c[i][5]*c[j][5]
                        v_4 = c[i][9]*c[j][1] +c[i][1]*c[j][9] +c[i][5]*c[j][5] +3*c[i][7]*c[j][2] +3*c[i][2]*c[j][7] +4*c[i][4]*c[j][4]
                        v_5 = 2*c[i][8]*c[j][1] +2*c[i][1]*c[j][8] +c[i][5]*c[j][3] +c[i][3]*c[j][5] +2*c[i][2]*c[j][9] +2*c[i][9]*c[j][2] +c[i][5]*c[j][4] +c[i][4]*c[j][5]
                        v_6 = 6*c[i][3]*c[j][6] +6*c[i][6]*c[j][3] +c[i][5]*c[j][8] +c[i][8]*c[j][5]
                        v_7 = c[i][5]*c[j][9] +c[i][9]*c[j][5] +6*c[i][4]*c[j][7] +6*c[i][7]*c[j][4]
                        v_8 = 3*c[i][6]*c[j][5] +3*c[i][5]*c[j][6] +4*c[i][3]*c[j][8] +4*c[i][8]*c[j][3] +2*c[i][8]*c[j][4] +2*c[i][4]*c[j][8] +2*c[i][5]*c[j][9] +2*c[i][9]*c[j][5]
                        v_9 = 3*c[i][7]*c[j][5] +3*c[i][5]*c[j][7] +4*c[i][4]*c[j][9]+ 4*+c[i][9]*c[j][4] +2*c[i][8]*c[j][5] +2*c[i][5]*c[j][8] +2*c[i][3]*c[j][9] +2*c[i][9]*c[j][3]
                        vec = np.array([v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9])
                        for numb in range(10):## integration over the triangles, probably better to do analytically
                            f = lambda x,y: x**(power_vec[numb][0])*y**(power_vec[numb][1])
                            if i%2==0:##upward oriented
                                val = integrate.dblquad(f, self.mesh.triangles[k].x_2[0], self.mesh.triangles[k].x_2[0]+self.mesh.h,
                                                                                     lambda x: x-self.mesh.triangles[k].x_2[0]+self.mesh.triangles[k].x_2[1]-self.mesh.h, self.mesh.triangles[k].x_2[1])[0]
                                ElementMatrix[k][i][j]+=vec[numb]*val
                            else:##downward oriented
                                ElementMatrix[k][i][j]+=vec[numb]*integrate.dblquad(f, self.mesh.triangles[k].x_0[0]-self.mesh.h, self.mesh.triangles[k].x_0[0],
                                                                                    self.mesh.triangles[k].x_0[1],lambda x: x-self.mesh.triangles[k].x_0[0]+self.mesh.triangles[k].x_0[1]+self.mesh.h)[0]
                    else:
                        ElementMatrix[k][i][j] = ElementMatrix[k][j][i]
        return ElementMatrix
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
            triangles.append(Triangle([v1.coordinates, v3.coordinates, v2.coordinates], [v1.global_number, v3.global_number, v2.global_number], loopcounter)) ##lower triangle
            triangles.append(Triangle([v4.coordinates, v3.coordinates, v1.coordinates], [v4.global_number, v3.global_number, v1.global_number], loopcounter+1)) ## upper triangle
            loopcounter+=2
    vertices = vertices.reshape((len(x)*len(y)))
    return Mesh(vertices, triangles, h)
def main():
    mesh = generateMesh_UnitSquare(0.25)
    soln = PoissonZ3Solver(mesh)
    #soln.IntegrationMatrixLHS()
    print(mesh.triangles[2].VertexNumbers)
if __name__=="__main__":
    main()
