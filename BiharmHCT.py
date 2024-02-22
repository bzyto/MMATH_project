import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from PoissonLinearApprox import gaussian_quad
import matplotlib.tri as mtri

class Triangle:
    def __init__(self, vertices, VertexNumbers =0, GlobalNumber = 0):
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
    def EdgeMidpoints(self):
            # Compute the midpoints
        midpoint1 = [(self.x_0[0] + self.x_1[0]) / 2, (self.x_0[1] + self.x_1[1]) / 2]
        midpoint2 = [(self.x_1[0] + self.x_2[0]) / 2, (self.x_1[1] + self.x_2[1]) / 2]
        midpoint3 = [(self.x_2[0] + self.x_0[0]) / 2, (self.x_2[1] + self.x_0[1]) / 2]

        midpoints = np.array([midpoint1, midpoint2, midpoint3])
        return midpoints
    def OuterUnitNormal(self):
        ###MAYBE WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        vectors = np.array([
            [self.x_0[1]-self.x_1[1], self.x_1[0]-self.x_0[0]]/np.linalg.norm([self.x_0[1]-self.x_1[1], self.x_1[0]-self.x_0[0]]),
            [self.x_1[1]-self.x_2[1], self.x_2[0]-self.x_1[0]]/np.linalg.norm([self.x_1[1]-self.x_2[1], self.x_2[0]-self.x_1[0]]),
            [self.x_2[1]-self.x_0[1], self.x_0[0]-self.x_2[0]]/np.linalg.norm([self.x_2[1]-self.x_0[1], self.x_0[0]-self.x_2[0]])
        ])
        ### here we dont know if these vectors point inwards or outwards ie we may have to multiply by -1
        ### we deduce the sign by computing the determinant of the following matrix
        mat = np.array([[vectors[0][0], vectors[0][1], 1], [vectors[1][0], vectors[1][1], 1], [vectors[2][0], vectors[2][1], 1]])
        if np.linalg.det(mat)>0:
            return -vectors
        return vectors
    def CreateMacroTriangles(self):
        macro_0 = Triangle([self.x_0, self.x_1, self.COM()])
        macro_1 = Triangle([self.x_1, self.x_2, self.COM()])
        macro_2 =Triangle([self.x_2, self.x_0, self.COM()])
        return np.array([macro_0, macro_1, macro_2])
    def LocalCubic(self):
        ###calculate the normals we want of the macro triangles
        ### find coordinates of macro triangles
        macrotrigs = self.CreateMacroTriangles()
        normal_at_x0COM = macrotrigs[0].OuterUnitNormal()[2]
        normal_at_x1COM = macrotrigs[1].OuterUnitNormal()[2]
        normal_at_x2COM = macrotrigs[2].OuterUnitNormal()[2]
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
        def nablaphi(vec):
            return np.array([phi_x(vec), phi_y(vec)])
        midpoint_x_0COM = 0.5 * (self.x_0+self.COM())
        midpoint_x_1COM = 0.5 * (self.x_1+self.COM())
        midpoint_x_2COM = 0.5 * (self.x_2 + self.COM())
        LHS = np.array([
            np.concatenate([phi(self.x_0), np.zeros(20)]),  # 0
            np.concatenate([phi_x(self.x_0), np.zeros(20)]),  # 1
            np.concatenate([phi_y(self.x_0), np.zeros(20)]),  # 2
            np.concatenate([phi(self.x_1), np.zeros(20)]),  # 3
            np.concatenate([phi_x(self.x_1), np.zeros(20)]),  # 4
            np.concatenate([phi_y(self.x_1), np.zeros(20)]),  # 5
            np.concatenate([phi(self.COM()), -phi(self.COM()), np.zeros(10)]),  # 6
            np.concatenate([np.dot(normal_at_x1COM, nablaphi(midpoint_x_1COM)), -np.dot(normal_at_x1COM, nablaphi(midpoint_x_1COM)), np.zeros(10)]),  # 7
            np.concatenate([np.dot(normal_at_x0COM, nablaphi(midpoint_x_0COM)),np.zeros(10), -np.dot(normal_at_x0COM, nablaphi(midpoint_x_0COM))]),  # 8
            np.concatenate([self.OuterUnitNormal()[0][0]*phi_x(1/2 * (self.x_0+self.x_1))+
                             self.OuterUnitNormal()[0][1]*phi_y(1/2 * (self.x_0+self.x_1)), np.zeros(20)]),  # 9
            np.concatenate([np.zeros(10), phi(self.x_1), np.zeros(10)]),  # 10
            np.concatenate([np.zeros(10), phi_x(self.x_1), np.zeros(10)]),  # 11
            np.concatenate([np.zeros(10), phi_y(self.x_1), np.zeros(10)]),  # 12
            np.concatenate([np.zeros(10), phi(self.x_2), np.zeros(10)]),  # 13
            np.concatenate([np.zeros(10), phi_x(self.x_2), np.zeros(10)]),  # 14
            np.concatenate([np.zeros(10), phi_y(self.x_2), np.zeros(10)]),  # 15
            np.concatenate([phi_x(self.COM()), -phi_x(self.COM()), np.zeros(10)]),  # 16
            np.concatenate([np.zeros(10), phi_x(self.COM()), -phi_x(self.COM())]),  # 17
            np.concatenate([phi_y(self.COM()), -phi_y(self.COM()), np.zeros(10)]),  # 18
            np.concatenate([np.zeros(10), phi_y(self.COM()), -phi_y(self.COM())]),  # 19
            np.concatenate([np.zeros(10), np.dot(normal_at_x2COM, nablaphi(midpoint_x_2COM)), -np.dot(normal_at_x2COM, nablaphi(midpoint_x_2COM))]),  # 20
            np.concatenate([np.zeros(10), self.OuterUnitNormal()[1][0]*phi_x(1/2 * (self.x_1+self.x_2))+
                             self.OuterUnitNormal()[1][1]*phi_y(1/2 * (self.x_1+self.x_2)), np.zeros(10)]),  # 21
            np.concatenate([np.zeros(20), phi(self.x_2)]),  # 22
            np.concatenate([np.zeros(20), phi_x(self.x_2)]),  # 23
            np.concatenate([np.zeros(20), phi_y(self.x_2)]),  # 24
            np.concatenate([np.zeros(20), phi(self.x_0)]),  # 25
            np.concatenate([np.zeros(20), phi_x(self.x_0)]),  # 26
            np.concatenate([np.zeros(20), phi_y(self.x_0)]),  # 27
            np.concatenate([phi(self.COM()), np.zeros(10), -phi(self.COM())]),  # 28
            np.concatenate([np.zeros(20), self.OuterUnitNormal()[2][0]*phi_x(1/2 * (self.x_0+self.x_2))+
                             self.OuterUnitNormal()[2][1]*phi_y(1/2 * (self.x_0+self.x_2))])  # 29
        ])
        pairs = [[0,25], [1,26], [2, 27], [3, 10], [4, 11], [5, 12], [13, 22], [14, 23], [15, 24], [9], [21], [-1]]  # enforcing the correct conditions on shape functions
        sols = []
        for i in range(12):
            RHS = np.zeros(30)
            for index in pairs[i]:
                RHS[index] = 1
            sols.append(np.linalg.solve(LHS, RHS))
        return sols
class Vertex:
    def __init__(self, coordinates, global_number, boundary, wb = None, val = None, wd=None):
        self.coordinates = coordinates
        self.global_number = global_number
        self.boundary = boundary
        self.which_boundary = wb
        self.val = val
        self.wd = wd #which derivative
    def __str__(self):
        return "Vertex with coordinates " +str(self.coordinates)+" and global number " +str(self.global_number)
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
class PoissonHCT:
    def __init__(self, mesh, bc = 0):
        self.mesh = mesh
        self.bc = bc
    def ElementIntegrationLHS(self):## vectorized by chatgpt
        n_triangles = len(self.mesh.triangles)
        ElementMatrix = np.zeros((n_triangles, 12, 12))

        for k in range(n_triangles):
            allcoefs = np.array(self.mesh.triangles[k].LocalCubic())
            for macrotriangle in range(3):
                c = allcoefs[:, macrotriangle*10:macrotriangle*10+10]
                i, j = np.indices((12, 12))
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
                ElementMatrix[k] += np.where(mask, gaussian_quad(f, self.mesh.triangles[k].CreateMacroTriangles()[macrotriangle]), ElementMatrix[k].T)
                #Ensure symmetry
            ElementMatrix[k] = np.where(mask, ElementMatrix[k], ElementMatrix[k].T)
            np.savetxt("EL.txt", ElementMatrix[0], fmt = '%.2f')

        return ElementMatrix
    def ElementIntegrationRHS(self, func = lambda x,y: 1):
        func = lambda x,y : 2*np.pi**2*np.sin(x*np.pi)*np.sin(y*np.pi)
        #func = lambda x,y: np.exp(-x**2-y**2)
        n_triangles = len(self.mesh.triangles)
        ElementVector = np.zeros((n_triangles, 12))
        for n in range(n_triangles):
            allcoefs = np.array(self.mesh.triangles[n].LocalCubic()) #coefficient matrix
            for macrotriangle in range(3):
                c = allcoefs[:, macrotriangle*10:macrotriangle*10+10]
                f = lambda x,y: func(x,y)*(c[:, 0] + c[:, 1] * x + c[:, 2] * y + c[:, 3] * x**2 + c[:, 4] * y**2 + c[:, 5] * x * y + c[:, 6] * x**3 + c[:, 7] * y**3 + c[:, 8] * x**2 * y + c[:, 9] * x * y**2)
                ElementVector[n] += gaussian_quad(f, self.mesh.triangles[n].CreateMacroTriangles()[macrotriangle])
        return ElementVector
    def Assembly(self):
        ElementMatrix = self.ElementIntegrationLHS()
        ElementVector = self.ElementIntegrationRHS()
        ConnectivityMatrix = self.mesh.ConnectivityMatrix()
        matrixSize = int(np.size(self.mesh.vertices))
        LHS = np.zeros((matrixSize, matrixSize))
        RHS = np.zeros(matrixSize)
        ## loop as in https://eprints.maths.manchester.ac.uk/894/2/0-19-852868-X.pdf p.26
        for k in range(len(self.mesh.triangles)):
            for j in range(12):
                for i in range(12):
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
                #### a bit tricky part
                ## we set the directional boundary conditions to 0 on the counterpart boundary
                ## ie u_x = 0 on y = 0 y =1 
                ## u_y = 0 on x = 0 x = 1
                elif self.mesh.vertices[i].which_boundary == 'y':
                    if self.mesh.vertices[i].wd == 'x':
                        LHS[i, :]= 0
                        LHS[i][i] = 1
                        RHS[i] = 0
                elif self.mesh.vertices[i].which_boundary == 'x':
                    if self.mesh.vertices[i].wd == 'y':
                        LHS[i, :]= 0
                        LHS[i][i] = 1
                        RHS[i] = 0
        return LHS, RHS
    def Solve(self):
        LHS, RHS = self.Assembly()
        solution = sparse.linalg.spsolve(sparse.csr_matrix(LHS), RHS)##taking advantage of the sparsity of the matrix
        np.savetxt("solution.txt", solution, fmt = '%.2f')
        return solution
    def PlotSolution(self):
        solution = self.Solve()
        soln = []
        for index in range(len(solution)):
            if self.mesh.vertices[index].wd == 'val':
                soln.append(solution[index])
        soln = np.array(soln)
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
        plt.show()
class BiharmHCT(PoissonHCT):
    def ElementIntegrationLHS(self):
        pass

def generateMesh_UnitSquare(h = 0.2):
    x = y = np.linspace(0, 1, int(2/h)+1)
    x_grid, y_grid = np.meshgrid(x, y)
    vertices = []
    triangles = []
    x_vertices = []
    y_vertices = []
    midpoints_dict = dict()
    midpoints = []
    loopcounter = 0
    for i in range(len(x_grid)):
        for j in range(len(x_grid)):
            boundary = False
            wb = None
            if x_grid[i][j] == 0 or x_grid[i][j] == 1 or y_grid[i][j] == 0 or y_grid[i][j] == 1:
                boundary = True
                if x_grid[i][j] == 0 or x_grid[i][j] == 1:
                    wb = 'x'
                else:
                    wb = 'y'
            if i%2 == 0 and j%2==0:
                vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter, boundary, wd = 'val'))
                x_vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter+1, boundary, wb, wd = 'x'))
                y_vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter+2, boundary, wb, wd = 'y'))
                loopcounter+=3
            else:
                midpoints.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter, boundary, wb, wd = 'norm'))
                midpoints_dict[(np.round(x_grid[i][j], 10), np.round(y_grid[i][j], 10))]= loopcounter
                loopcounter+=1
    total_number_of_vertices = loopcounter
    x = x[::2]
    y = y[::2]
    vertices = np.array(vertices)
    vertices = vertices.reshape(len(x), len(y))
    x_vertices = np.array(x_vertices)
    x_vertices = x_vertices.reshape(len(x), len(y))
    y_vertices = np.array(y_vertices)
    y_vertices = y_vertices.reshape(len(x), len(y))
    loopcounter = 0
    def get_tuple(x,y):
        a = (x.coordinates[0] + y.coordinates[0])/2
        b = (x.coordinates[1] + y.coordinates[1])/2
        return (np.round(a, 10),np.round(b, 10))
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
            triangles.append(Triangle([v2.coordinates, v1.coordinates, v3.coordinates], [v2.global_number, v2_x.global_number, v2_y.global_number, v1.global_number, v1_x.global_number, v1_y.global_number,
                                                                                          v3.global_number, v3_x.global_number, v3_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v2, v1)), midpoints_dict.get(get_tuple(v1, v3)), 
                                                                                          midpoints_dict.get(get_tuple(v2, v3))], loopcounter)) ##lower triangle
            triangles.append(Triangle([v1.coordinates, v4.coordinates, v3.coordinates], [v1.global_number, v1_x.global_number, v1_y.global_number, v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          v3.global_number, v3_x.global_number, v3_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v1, v4)), midpoints_dict.get(get_tuple(v4, v3)), midpoints_dict.get(get_tuple(v3, v1))], loopcounter+1)) ##upper triangle
            loopcounter+=2
    all_but_coms = []
    vertices = vertices.reshape(len(x)*len(y))
    x_vertices = x_vertices.reshape(len(x)*len(y))
    y_vertices = y_vertices.reshape(len(x)*len(y))
    for i in range(len(vertices)):
        all_but_coms.append(vertices[i])
        all_but_coms.append(x_vertices[i])
        all_but_coms.append(y_vertices[i])
    nomids = len(all_but_coms)
    for mid in midpoints:
        all_but_coms.append(mid)
    all_but_coms.sort(key = lambda x: x.global_number)
    return Mesh(all_but_coms, triangles, h, nomids)

mesh = generateMesh_UnitSquare(1/32)
sol = PoissonHCT(mesh)
sol.PlotSolution()
