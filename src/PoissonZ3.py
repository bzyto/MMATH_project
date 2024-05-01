import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import cProfile
import matplotlib.tri as mtri
import time
from PoissonLinearApprox import gaussian_quad
plt.rcParams['text.usetex'] = True #for nice plots
class Triangle:
    def __init__(self, vertices, VertexNumbers, GlobalNumber = 0):
        self.x_0 = np.array(vertices[0])
        self.x_1 = np.array(vertices[1])
        self.x_2 = np.array(vertices[2])
        self.vertices = vertices
        self.GlobalNumber = GlobalNumber
        self.VertexNumbers = VertexNumbers

    def area(self):
        return 0.5 * np.abs(np.cross(self.x_1 - self.x_0, self.x_2 - self.x_0))
    def __str__(self):
        return "Triangle with vertices \n" + str(self.x_0) + "\n" + str(self.x_1) + "\n" + str(self.x_2) + "\n And global number " + str(self.GlobalNumber) + "\n and vertices numbers:"+str(self.VertexNumbers)
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
        # row_1 = np.array([1, self.x_0[0], self.x_0[1], self.x_0[0]**2, self.x_0[1]**2, self.x_0[0]*self.x_0[1], self.x_0[0]**3, self.x_0[1]**3, self.x_0[0]**2*self.x_0[1], self.x_0[0]*self.x_0[1]**2]) ##phi(x_0, y_0)
        # row_2 = np.array([0, 1, 0, 2*self.x_0[0], 0, self.x_0[1], 3*self.x_0[0]**2, 0, 2*self.x_0[0]*self.x_0[1], self.x_0[1]**2]) ## d/dx phi(x_0, y_0)
        # row_3 = np.array([0, 0, 1, 0, 2*self.x_0[1], self.x_0[0], 0, 3*self.x_0[1]**2, self.x_0[0]**2, 2*self.x_0[0]*self.x_0[1]]) ## d/dy phi(x_0, y_0)
        # row_4 = np.array([1, self.x_1[0], self.x_1[1], self.x_1[0]**2, self.x_1[1]**2, self.x_1[0]*self.x_1[1], self.x_1[0]**3, self.x_1[1]**3, self.x_1[0]**2*self.x_1[1], self.x_1[0]*self.x_1[1]**2]) ##phi(x_1, y_1)
        # row_5 = np.array([0, 1, 0, 2*self.x_1[0], 0, self.x_1[1], 3*self.x_1[0]**2, 0, 2*self.x_1[0]*self.x_1[1], self.x_1[1]**2])
        # row_6 = np.array([0, 0, 1, 0, 2*self.x_1[1], self.x_1[0], 0, 3*self.x_1[1]**2, self.x_1[0]**2, 2*self.x_1[0]*self.x_1[1]]) 
        # row_7 = np.array([1, self.x_2[0], self.x_2[1], self.x_2[0]**2, self.x_2[1]**2, self.x_2[0]*self.x_2[1], self.x_2[0]**3, self.x_2[1]**3, self.x_2[0]**2*self.x_2[1], self.x_2[0]*self.x_2[1]**2])
        # row_8 = np.array([0, 1, 0, 2*self.x_2[0], 0, self.x_2[1], 3*self.x_2[0]**2, 0, 2*self.x_2[0]*self.x_2[1], self.x_2[1]**2])
        # row_9 = np.array([0, 0, 1, 0, 2*self.x_2[1], self.x_2[0], 0, 3*self.x_2[1]**2, self.x_2[0]**2, 2*self.x_2[0]*self.x_2[1]])
        # row_10 = np.array([1, self.COM()[0], self.COM()[1], self.COM()[0]**2, self.COM()[1]**2, self.COM()[0]*self.COM()[1], self.COM()[0]**3, self.COM()[1]**3, self.COM()[0]**2*self.COM()[1], self.COM()[0]*self.COM()[1]**2]) #phi at com 
        # LHS = np.array([row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10])
        # solution = []
        def phi(vec):
            x = vec[0]
            y = vec[1]
            return np.array([1, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])
    
        def phi_x(vec):
            x = vec[0]
            y = vec[1]
            return np.array([0, 1, 0, 2*x, 0, y, 3*x**2, 0, 2*x*y, y**2])
        
        def phi_y(vec):
            x = vec[0]
            y = vec[1]
            return np.array([0, 0, 1, 0, 2*y, x, 0, 3*y**2, x**2, 2*x*y])
        LHS = np.array([
            phi(self.x_0),
            phi_x(self.x_0),
            phi_y(self.x_0),
            phi(self.x_1),
            phi_x(self.x_1),
            phi_y(self.x_1),
            phi(self.x_2),
            phi_x(self.x_2),
            phi_y(self.x_2),
            phi(self.COM())
        ])
        solution = []
        for i in range(10):
            RHS = np.zeros(10)
            RHS[i] = 1
            solution.append(np.linalg.solve(LHS, RHS))
        return np.array(solution)

class Vertex:
    def __init__(self, coordinates, global_number, boundary, wb = None, val = None):
        self.coordinates = coordinates
        self.global_number = global_number
        self.boundary = boundary
        self.which_boundary = wb
        self.val = val
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
                # plt.text(triangle.x_0[0], triangle.x_0[1], 0)#str(triangle.VertexNumbers[0]))
                # plt.text(triangle.x_1[0], triangle.x_1[1], 1)#str(triangle.VertexNumbers[1]))
                # plt.text(triangle.x_2[0], triangle.x_2[1], 2)#str(triangle.VertexNumbers[2]))
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
    def MapSolution(self, solution):
        for index, value in enumerate(solution.flatten()):
            try:
                self.vertices[index].val = value
            except IndexError:
                return
class PoissonZ3Solver:
    def __init__(self, mesh, bc=0):
        self.mesh = mesh
        self.bc = bc
    # def ElementIntegrationLHS(self):## my implementation
    #     n_triangles = len(self.mesh.triangles)
    #     ElementMatrix = np.zeros((n_triangles, 10, 10))
    #     for k in range(n_triangles):
    #         c = self.mesh.triangles[k].LocalCubic() #coefficient matrix
    #         for i in range(10):
    #             for j in range(10):
    #                 ## create a vector with c^i_kc^j_l coefficients as outlined in project
    #                 if j>=i:#taking advantage of the symmetry
    #                     v_0 = c[i][1]*c[j][1]+ c[i][2]*c[j][2]#1
    #                     v_1 = 2*c[i][1]*c[j][3]+c[i][2]*c[j][5]+2*c[i][3]*c[j][1]+ c[i][5]*c[j][2]#x
    #                     v_2 = c[i][1]*c[j][5]+2*c[i][2]*c[j][4]+ 2*c[i][4]*c[j][2]+ c[i][5]*c[j][1]#y
    #                     v_3 = 3*c[i][1]*c[j][6]+c[i][2]*c[j][8]+4*c[i][3]*c[j][3]+c[i][5]*c[j][5]+ 3*c[i][6]*c[j][1] + c[i][8]*c[j][2]#x^2
    #                     v_4 = c[i][1]*c[j][9] + 3*c[i][2]*c[j][7] + 4*c[i][4]*c[j][4]+ c[i][5]*c[j][5] + 3*c[i][7]*c[j][2] + c[i][9]*c[j][1]#y^2
    #                     v_5 =2*c[i][1]*c[j][8]+2*c[i][2]*c[j][9]+2*c[i][3]*c[j][5] + 2*c[i][4]*c[j][5]+ 2*c[i][5]*c[j][3] + 2*c[i][5]*c[j][4] + 2*c[i][8]*c[j][1]+2*c[i][9]*c[j][2]#xy
    #                     v_6 = 6*c[i][3]*c[j][6]+ c[i][5]*c[j][8] + 6*c[i][6]*c[j][3] + c[i][8]*c[j][5] #x^3
    #                     v_7 = 6*c[i][4]*c[j][7] + c[i][5]*c[j][9] + 6*c[i][7]*c[j][4] + c[i][9]*c[j][5]#y^3
    #                     v_8 = 4*c[i][3]*c[j][8]+ 2*c[i][4]*c[j][8]+ 3*c[i][5]*c[j][6] + 2*c[i][5]*c[j][9] + 3*c[i][6]*c[j][5] + 4*c[i][8]*c[j][3] + 2*c[i][8]*c[j][4] + 2*c[i][9]*c[j][5]#x^2y
    #                     v_9 = 2*c[i][3]*c[j][9]+ 4*c[i][4]*c[j][9] + 3*c[i][5]*c[j][7] + 2*c[i][5]*c[j][8]+ 3*c[i][7]*c[j][5] + 2*c[i][9]*c[j][3] + 4*c[i][9]*c[j][4] + 2*c[i][8]*c[j][5]#xy^2
    #                     v_10 = 9*c[i][6]*c[j][6] + c[i][8]*c[j][8]#x^4
    #                     v_11 = 9*c[i][7]*c[j][7]+ c[i][9]*c[j][9]#y^4
    #                     v_12 = 6*c[i][6]*c[j][8] + 6*c[i][8]*c[j][6] + 2*c[i][8]*c[j][9] + 2*c[i][9]*c[j][8]#x^3y
    #                     v_13 = 6*c[i][7]*c[j][9]+ 6*c[i][9]*c[j][7]+ 2*c[i][9]*c[j][8] + 2*c[i][8]*c[j][9]#xy^3
    #                     v_14 = 3*c[i][6]*c[j][9]+ 3*c[i][7]*c[j][8] + 3*c[i][9]*c[j][6] + 3*c[i][8]*c[j][7] + 4*c[i][8]*c[j][8]+4*c[i][9]*c[j][9]#x^2y^2
    #                     f = lambda x,y: v_0+v_1*x+v_2*y+v_3*x**2+v_4*y**2+v_5*x*y+v_6*x**3+v_7*y**3+v_8*x**2*y+v_9*x*y**2+v_10*x**4+v_11*y**4+v_12*x**3*y+v_13*x*y**3+v_14*x**2*y**2
    #                     ElementMatrix[k][i][j] = gaussian_quad(f, self.mesh.triangles[k])
    #                 else:
    #                     ElementMatrix[k][i][j] = ElementMatrix[k][j][i]
    #     return ElementMatrix
    def ElementIntegrationLHS(self):## vectorized by chatgpt
        n_triangles = len(self.mesh.triangles)
        ElementMatrix = np.zeros((n_triangles, 10, 10))

        for k in range(n_triangles):
            c = self.mesh.triangles[k].LocalCubic()
            i, j = np.indices((10, 10))
            mask = j >= i

            v_0 = np.outer(c[:, 1], c[:, 1]) + np.outer(c[:, 2], c[:, 2])  # 1
            v_1 = 2 * np.outer(c[:, 1], c[:, 3]) + np.outer(c[:, 2], c[:, 5]) + 2 * np.outer(c[:, 3], c[:, 1]) + np.outer(c[:, 5], c[:, 2])  # x
            v_2 = np.outer(c[:, 1], c[:, 5]) + 2 * np.outer(c[:, 2], c[:, 4]) + 2 * np.outer(c[:, 4], c[:, 2]) + np.outer(c[:, 5], c[:, 1])  # y
            v_3 = 3 * np.outer(c[:, 1], c[:, 6]) + np.outer(c[:, 2], c[:, 8]) + 4 * np.outer(c[:, 3], c[:, 3]) + np.outer(c[:, 5], c[:, 5]) + 3 * np.outer(c[:, 6], c[:, 1]) + np.outer(c[:, 8], c[:, 2])  # x^2
            v_4 = np.outer(c[:, 1], c[:, 9]) + 3 * np.outer(c[:, 2], c[:, 7]) + 4 * np.outer(c[:, 4], c[:, 4]) + np.outer(c[:, 5], c[:, 5]) + 3 * np.outer(c[:, 7], c[:, 2]) + np.outer(c[:, 9], c[:, 1])  # y^2
            v_5 = 2 * np.outer(c[:, 1], c[:, 8]) + 2 * np.outer(c[:, 2], c[:, 9]) + 2 * np.outer(c[:, 3], c[:, 5]) + 2 * np.outer(c[:, 4], c[:, 5]) + 2 * np.outer(c[:, 5], c[:, 3]) + 2 * np.outer(c[:, 5], c[:, 4]) + 2 * np.outer(c[:, 8], c[:, 1]) + 2 * np.outer(c[:, 9], c[:, 2])  # xy
            v_6 = 6 * np.outer(c[:, 3], c[:, 6]) + np.outer(c[:, 5], c[:, 8]) + 6 * np.outer(c[:, 6], c[:, 3]) + np.outer(c[:, 8], c[:, 5])  # x^3
            v_7 = 6 * np.outer(c[:, 4], c[:, 7]) + np.outer(c[:, 5], c[:, 9]) + 6 * np.outer(c[:, 7], c[:, 4]) + np.outer(c[:, 9], c[:, 5])  # y^3
            v_8 = 4 * np.outer(c[:, 3], c[:, 8]) + 2 * np.outer(c[:, 4], c[:, 8]) + 3 * np.outer(c[:, 5], c[:, 6]) + 2 * np.outer(c[:, 5], c[:, 9]) + 3 * np.outer(c[:, 6], c[:, 5]) + 4 * np.outer(c[:, 8], c[:, 3]) + 2 * np.outer(c[:, 8], c[:, 4]) + 2 * np.outer(c[:, 9], c[:, 5])  # x^2y
            v_9 = 2 * np.outer(c[:, 3], c[:, 9]) + 4 * np.outer(c[:, 4], c[:, 9]) + 3 * np.outer(c[:, 5], c[:, 7]) + 2 * np.outer(c[:, 5], c[:, 8]) + 3 * np.outer(c[:, 7], c[:, 5]) + 2 * np.outer(c[:, 9], c[:, 3]) + 4 * np.outer(c[:, 9], c[:, 4]) + 2 * np.outer(c[:, 8], c[:, 5])  # xy^2
            v_10 = 9 * np.outer(c[:, 6], c[:, 6]) + np.outer(c[:, 8], c[:, 8])  # x^4
            v_11 = 9 * np.outer(c[:, 7], c[:, 7]) + np.outer(c[:, 9], c[:, 9])  # y^4
            v_12 = 6 * np.outer(c[:, 6], c[:, 8]) + 6 * np.outer(c[:, 8], c[:, 6]) + 2 * np.outer(c[:, 8], c[:, 9]) + 2 * np.outer(c[:, 9], c[:, 8])  # x^3y
            v_13 = 6 * np.outer(c[:, 7], c[:, 9]) + 6 * np.outer(c[:, 9], c[:, 7]) + 2 * np.outer(c[:, 9], c[:, 8]) + 2 * np.outer(c[:, 8], c[:, 9])  # xy^3
            v_14 = 3 * np.outer(c[:, 6], c[:, 9]) + 3 * np.outer(c[:, 7], c[:, 8]) + 3 * np.outer(c[:, 9], c[:, 6]) + 3 * np.outer(c[:, 8], c[:, 7]) + 4 * np.outer(c[:, 8], c[:, 8]) + 4 * np.outer(c[:, 9], c[:, 9])  # x^2y^2
            f = lambda x,y :v_0 + v_1 * x + v_2 * y + v_3 * x**2 + v_4 * y**2 + v_5 * x*y + v_6 * x**3 + v_7 * y**3 + v_8 * x**2*y + v_9 * x*y**2 + v_10 * x**4 + v_11 * y**4 + v_12 * x**3*y + v_13 * x*y**3 + v_14 * x**2*y**2
            ElementMatrix[k] = np.where(mask, gaussian_quad(f, self.mesh.triangles[k]), ElementMatrix[k].T)

            # Ensure symmetry
            ElementMatrix[k] = np.where(mask, ElementMatrix[k], ElementMatrix[k].T)
        return ElementMatrix

    def TestIntegrationLHS(self):
        ElementMatrix = self.ElementIntegrationLHS()
        for k in range(len(self.mesh.triangles)):
            for i in range(10):
                if ElementMatrix[k][i][i]<0:
                    print(ElementMatrix[k][i][i])
                    print(k, i)
                    return False
        return True
            
    def ElementIntegrationRHS(self, func = lambda x,y: 1):
        func = lambda x,y : 2*np.pi**2*np.sin(x*np.pi)*np.sin(y*np.pi)
        #func = lambda x,y: np.exp(-x**2-y**2)
        n_triangles = len(self.mesh.triangles)
        ElementVector = np.zeros((n_triangles, 10))
        for n in range(n_triangles):
            c = self.mesh.triangles[n].LocalCubic() #coefficient matrix
            f = lambda x,y: func(x,y)*(c[:, 0] + c[:, 1] * x + c[:, 2] * y + c[:, 3] * x**2 + c[:, 4] * y**2 + c[:, 5] * x * y + c[:, 6] * x**3 + c[:, 7] * y**3 + c[:, 8] * x**2 * y + c[:, 9] * x * y**2)
            ElementVector[n] = gaussian_quad(f, self.mesh.triangles[n])
        return ElementVector
    def Assembly(self):
        ElementMatrix = self.ElementIntegrationLHS()
        ElementVector = self.ElementIntegrationRHS()
        ConnectivityMatrix = self.mesh.ConnectivityMatrix()
        matrixSize = int(np.size(self.mesh.vertices)+np.size(self.mesh.triangles))
        LHS = np.zeros((matrixSize, matrixSize))
        RHS = np.zeros(matrixSize)
        ## loop as in https://eprints.maths.manchester.ac.uk/894/2/0-19-852868-X.pdf p.26
        for k in range(len(self.mesh.triangles)):
            for j in range(10):
                for i in range(10):
                    LHS[int(ConnectivityMatrix[k][i])][int(ConnectivityMatrix[k][j])]+=ElementMatrix[k][i][j]
                RHS[int(ConnectivityMatrix[k][j])]+=ElementVector[k][j]
        ## at this point, the galerkin matrix is of size n by n where n is the total number of vertices
        ## we have enforce the boundary conditions on the system
        for i in range(len(self.mesh.vertices)):
            if self.mesh.vertices[i].boundary:
                if self.mesh.vertices[i].which_boundary == None:
                    LHS[i, :]= 0
                    LHS[i][i] = 1
                    RHS[i] = 0
                ### a bit tricky part
                # we set the directional boundary conditions to 0 on the counterpart boundary
                # ie u_x = 0 on y = 0 y =1 
                # u_y = 0 on x = 0 x = 1
                elif self.mesh.vertices[i].which_boundary == 'y':
                    if i%3 == 1:
                        LHS[i, :]= 0
                        LHS[i][i] = 1
                        RHS[i] = 0
                elif self.mesh.vertices[i].which_boundary == 'x':
                    if i%3 == 2:
                        LHS[i, :]= 0
                        LHS[i][i] = 1
                        RHS[i] = 0

        return LHS, RHS
    def Solve(self):
        LHS, RHS = self.Assembly()
        #matrix_to_binary_image(LHS)
        solution = sparse.linalg.spsolve(sparse.csr_matrix(LHS), RHS)##taking advantage of the sparsity of the matrix
        return solution
    def PlotSolution(self):
        solution = self.Solve()
        solution = solution[0:len(self.mesh.vertices)]
        soln = []
        for i in range(len(solution)):
            if i%3==0:
                soln.append(solution[i])
        soln = np.array(soln)
        np.savetxt("Z3sol.txt", soln, fmt='%.4f')
        print(np.max(soln))
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
        ax.set_title(fr'$-\nabla u = 2\pi^2\sin(\pi x)\sin(\pi y)$ solution with $h = {self.mesh.h}$, and $u = 0$ on $\partial \Omega$')
        plt.show()
    def Plot_and_Save(self):
        solution = self.Solve()
        solution = solution[0:len(self.mesh.vertices)]
        soln = []
        for i in range(len(solution)):
            if i%3==0:
                soln.append(solution[i])
        soln = np.array(soln)
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
        ax.set_title(fr'$-\Delta u = 2\pi^2\sin(\pi x)\sin(\pi y)$ and $u = 0$ on $\partial \Omega$, solution with $h = {self.mesh.h}$')
        plt.savefig(f'figures/poisson_z3_{self.mesh.h}__.jpg', dpi=300)
def generateMesh_UnitSquare(h = 0.2):
    x = y = np.linspace(0, 1, int(1/h)+1)
    x_grid, y_grid = np.meshgrid(x, y)
    vertices = []
    triangles = []
    x_vertices = []
    y_vertices = []
    loopcounter = 0
    for i in range(len(x_grid)):
        for j in range(len(x_grid)):
            boundary = False
            if x_grid[i][j] == 0 or x_grid[i][j] == 1 or y_grid[i][j] == 0 or y_grid[i][j] == 1:
                boundary = True
                if x_grid[i][j] == 0 or x_grid[i][j] == 1:
                    wb = 'x'
                else:
                    wb = 'y'
            vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter, boundary))
            x_vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter+1, boundary, wb))
            y_vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter+2, boundary, wb))
            loopcounter+=3
    total_number_of_vertices = loopcounter
    vertices = np.array(vertices)
    vertices = vertices.reshape(len(x), len(y))
    x_vertices = np.array(x_vertices)
    x_vertices = x_vertices.reshape(len(x), len(y))
    y_vertices = np.array(y_vertices)
    y_vertices = y_vertices.reshape(len(x), len(y))
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
            v1_x = x_vertices[i][j]
            v2_x = x_vertices[i+1][j]
            v3_x = x_vertices[i+1][j+1]
            v4_x = x_vertices[i][j+1]
            v1_y = y_vertices[i][j]
            v2_y = y_vertices[i+1][j]
            v3_y = y_vertices[i+1][j+1]
            v4_y = y_vertices[i][j+1]
            triangles.append(Triangle([v1.coordinates, v2.coordinates, v3.coordinates], [v1.global_number, v1_x.global_number, v1_y.global_number, v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          v3.global_number, v3_x.global_number, v3_y.global_number, total_number_of_vertices+loopcounter], loopcounter)) ##lower triangle
            triangles.append(Triangle([v4.coordinates, v1.coordinates, v3.coordinates], [v4.global_number, v4_x.global_number, v4_y.global_number, v1.global_number, v1_x.global_number, v1_y.global_number,
                                                                                          v3.global_number, v3_x.global_number, v3_y.global_number, total_number_of_vertices+loopcounter+1], loopcounter+1)) ##upper triangle
            loopcounter+=2
    all_but_coms = []
    vertices = vertices.reshape(len(x)*len(y))
    x_vertices = x_vertices.reshape(len(x)*len(y))
    y_vertices = y_vertices.reshape(len(x)*len(y))
    for i in range(len(vertices)):
        all_but_coms.append(vertices[i])
        all_but_coms.append(x_vertices[i])
        all_but_coms.append(y_vertices[i])
    return Mesh(all_but_coms, triangles, h)
def matrix_to_binary_image(matrix):
    # Create a binary matrix where 1 represents nonzero entries
    binary_matrix = np.where(matrix != 0, 1, 0)

    # Plot the binary image
    plt.imshow(binary_matrix, cmap='binary', interpolation='nearest')
    plt.show()
import time
def u_exact(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)
def main():
    mesh = generateMesh_UnitSquare(1/32)
    solver = PoissonZ3Solver(mesh)
    solver.Plot_and_Save()
if __name__=="__main__":
    main()