import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.linalg import issymmetric
from scipy import sparse
from Elements import *
from mesh import *
from functions import *
plt.rcParams['text.usetex'] = True #for nice plots
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
                c = allcoefs[:,macrotriangle*10:macrotriangle*10+10]
                i, j = np.indices((12, 12))
                mask = j >= i

                v_0 = np.outer(c[:,1], c[:,1]) + np.outer(c[:, 2], c[:, 2])  # 1
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
            # pairs = [[0,9], [1,9], [2,9], [3, 10], [4, 10], [5,10], [6, 11], [7,11], [8,11], [9,10], [9, 11], [10,11]]
            # for pair in pairs:
            #     ElementMatrix[k][pair[0]][pair[1]] = 0
            ElementMatrix[k] = np.where(mask, ElementMatrix[k], ElementMatrix[k].T)
        return ElementMatrix
    def ElementIntegrationRHS(self, func = lambda x,y: 1):
        #func = lambda x,y : 2*np.pi**2*np.sin(x*np.pi)*np.sin(y*np.pi)
        #func = lambda x,y: np.exp(-x**2-y**2)
        n_triangles = len(self.mesh.triangles)
        ElementVector = np.zeros((n_triangles, 12))
        for n in range(n_triangles):
            allcoefs = np.array(self.mesh.triangles[n].LocalCubic()) #coefficient matrix
            for macrotriangle in range(3):
                c = allcoefs[:, macrotriangle*10:macrotriangle*10+10]
                f = lambda x,y: func(x,y)*(c[:, 0] + c[:, 1] * x + c[:, 2] * y + c[:, 3] * x**2 + c[:, 4] * y**2 + c[:, 5] * x * y + c[:, 6] * x**3 + c[:, 7] * y**3 + c[:, 8] * x**2 * y + c[:, 9] * x * y**2)
                # ElementVector[n][zers[macrotriangle]] += gaussian_quad(f, self.mesh.triangles[n].CreateMacroTriangles()[macrotriangle])[zers[macrotriangle]]
                ElementVector[n] += gaussian_quad(f, self.mesh.triangles[n].CreateMacroTriangles()[macrotriangle])
        return ElementVector
    def Assembly(self):
        ElementMatrix = self.ElementIntegrationLHS()
        ElementVector = self.ElementIntegrationRHS()
        ConnectivityMatrix = self.mesh.ConnectivityMatrix()
        matrixSize = int(np.size(self.mesh.vertices))
        LHS = np.zeros((matrixSize, matrixSize))
        RHS = np.zeros(matrixSize)
        Idid = np.ones(matrixSize)
        ## loop as in https://eprints.maths.manchester.ac.uk/894/2/0-19-852868-X.pdf p.26
        for k in range(len(self.mesh.triangles)):
            marked = []
            for j in range(12):
                fix_j = 1
                if any([j == 9, j==10, j==11]):
                    fix_j = Idid[int(ConnectivityMatrix[k][j])]
                for i in range(12):
                    fix_i = 1
                    if any([i == 9, i==10, i==11]):
                        fix_i = Idid[int(ConnectivityMatrix[k][i])]
                    #  if check_elements_in_arrays([int(ConnectivityMatrix[k][j]), int(ConnectivityMatrix[k][i])], edges[0], edges[1], edges[2]):
                    LHS[int(ConnectivityMatrix[k][j])][int(ConnectivityMatrix[k][i])]+=fix_i * fix_j * ElementMatrix[k][j][i]
                RHS[int(ConnectivityMatrix[k][j])]+=fix_j * ElementVector[k][j]
            Idid[self.mesh.triangles[k].VertexNumbers[9:]] = -1 ##as in the mysterious matlab file
        ## at this point, the galerkin matrix is of size n by n where n is the total number of vertices
        ## we have enforce the boundary conditions on the system
        for i in range(len(self.mesh.vertices)):
            if self.mesh.vertices[i].boundary:
                if self.mesh.vertices[i].wd == 'val':
                    LHS[i, :]= 0
                    LHS[i][i] = 1
                    RHS[i] = 0
                ### a bit tricky part
                # we set the directional boundary conditions to 0 on the counterpart boundary
                # ie u_x = 0 on y = 0 y =1 
                # u_y = 0 on x = 0 x = 1
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
        # for i in range(len(self.mesh.triangles)):
        #     normal_vertices_nums = self.mesh.triangles[i].VertexNumbers[-3:]
        #     for index, numb in enumerate(normal_vertices_nums):
        #         if self.mesh.vertices[numb].boundary:
        #             LHS[numb,:] = 0
        #             LHS[numb][numb] = 1
        #             RHS[numb] = np.dot(
        #                 self.mesh.triangles[i].OuterUnitNormal()[index],
        #                   [np.pi*np.cos(np.pi*self.mesh.vertices[numb].coordinates[0])*np.sin(np.pi*self.mesh.vertices[numb].coordinates[1]),
        #                    np.pi*np.sin(np.pi*self.mesh.vertices[numb].coordinates[0])*np.cos(np.pi*self.mesh.vertices[numb].coordinates[1])]
        #             )

        return LHS, RHS
    def Solve(self):
        LHS, RHS = self.Assembly()
        solution = sparse.linalg.spsolve(sparse.csr_matrix(LHS), RHS)##taking advantage of the sparsity of the matrix
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
class BiharmHCT:
    def __init__(self, mesh, bc = 0):
        self.mesh = mesh
        self.bc = bc
    def ElementIntegrationLHS(self):
        n_triangles = len(self.mesh.triangles)
        ElementMatrix = np.zeros((n_triangles, 12, 12))

        for k in range(n_triangles):
            allcoefs = np.array(self.mesh.triangles[k].LocalCubic())
            for macrotriangle in range(3):
                c = allcoefs[:,macrotriangle*10:macrotriangle*10+10]
                i, j = np.indices((12, 12))
                mask = j >= i
                def laplace(x,y):
                    return 2*c[:,3]+ 6*c[:,6]*x+2*c[:,8]*y+2*c[:,4]+6*c[:,7]*y+2*c[:,9]*x
                def f(x,y):
                    return np.outer(laplace(x,y), laplace(x,y))
                ElementMatrix[k] += np.where(mask, gaussian_quad(f, self.mesh.triangles[k].CreateMacroTriangles()[macrotriangle]), ElementMatrix[k].T)
            #Ensure symmetry
            # pairs = [[0,9], [1,9], [2,9], [3, 10], [4, 10], [5,10], [6, 11], [7,11], [8,11], [9,10], [9, 11], [10,11]]
            # for pair in pairs:
            #     ElementMatrix[k][pair[0]][pair[1]] = 0
            ElementMatrix[k] = np.where(mask, ElementMatrix[k], ElementMatrix[k].T)
        return ElementMatrix
    def ElementIntegrationRHS(self, func = lambda x,y: 1):
        #func = lambda x,y : 2*np.pi**2*np.sin(x*np.pi)*np.sin(y*np.pi)
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
    def ElementIntegrationEigenRHS(self):
        n_triangles = len(self.mesh.triangles)
        ElementMatrix = np.zeros((n_triangles, 12, 12))
        for k in range(n_triangles):
            allcoefs = np.array(self.mesh.triangles[k].LocalCubic())
            for macrotriangle in range(3):
                c = allcoefs[:,macrotriangle*10:macrotriangle*10+10]
                i, j = np.indices((12, 12))
                mask = j >= i
                def f(x,y):
                    return np.outer((c[:, 0] + c[:, 1] * x + c[:, 2] * y + c[:, 3] * x**2 + c[:, 4] * y**2 + c[:, 5] * x * y + c[:, 6] * x**3 + c[:, 7] * y**3 + c[:, 8] * x**2 * y + c[:, 9] * x * y**2),
                                    (c[:, 0] + c[:, 1] * x + c[:, 2] * y + c[:, 3] * x**2 + c[:, 4] * y**2 + c[:, 5] * x * y + c[:, 6] * x**3 + c[:, 7] * y**3 + c[:, 8] * x**2 * y + c[:, 9] * x * y**2))
                ElementMatrix[k] += np.where(mask, gaussian_quad(f, self.mesh.triangles[k].CreateMacroTriangles()[macrotriangle]), ElementMatrix[k].T)
            ElementMatrix[k] = np.where(mask, ElementMatrix[k], ElementMatrix[k].T)
        return ElementMatrix
    def Assembly(self):
        ElementMatrix = self.ElementIntegrationLHS()
        ElementVector = self.ElementIntegrationRHS()
        ConnectivityMatrix = self.mesh.ConnectivityMatrix()
        matrixSize = int(np.size(self.mesh.vertices))
        LHS = np.zeros((matrixSize, matrixSize))
        RHS = np.zeros(matrixSize)
        Idid = np.ones(matrixSize)
        ## loop as in https://eprints.maths.manchester.ac.uk/894/2/0-19-852868-X.pdf p.26
        for k in range(len(self.mesh.triangles)):
            marked = []
            for j in range(12):
                fix_j = 1
                if any([j == 9, j==10, j==11]):
                    fix_j = Idid[int(ConnectivityMatrix[k][j])]
                for i in range(12):
                    fix_i = 1
                    if any([i == 9, i==10, i==11]):
                        fix_i = Idid[int(ConnectivityMatrix[k][i])]
                    #  if check_elements_in_arrays([int(ConnectivityMatrix[k][j]), int(ConnectivityMatrix[k][i])], edges[0], edges[1], edges[2]):
                    LHS[int(ConnectivityMatrix[k][j])][int(ConnectivityMatrix[k][i])]+=fix_i * fix_j * ElementMatrix[k][j][i]
                RHS[int(ConnectivityMatrix[k][j])]+=fix_j * ElementVector[k][j]
            Idid[self.mesh.triangles[k].VertexNumbers[9:]] = -1 ##as in the mysterious matlab file
        ## at this point, the galerkin matrix is of size n by n where n is the total number of vertices
        ## we have enforce the boundary conditions on the system
        for i in range(len(self.mesh.vertices)):
            if self.mesh.vertices[i].boundary:
                # if self.mesh.vertices[i].wd == 'val':
                LHS[i, :]= 0
                LHS[i][i] = 1
                RHS[i] = 0
                # ### a bit tricky part
                # # we set the directional boundary conditions to 0 on the counterpart boundary
                # # ie u_x = 0 on y = 0 y =1 
                # # u_y = 0 on x = 0 x = 1
                # if self.mesh.vertices[i].wd == 'norm':
                #     LHS[i, :]= 0
                #     LHS[i][i] = 1
                #     RHS[i] = 0
                # if self.mesh.vertices[i].which_boundary == 'y':
                #     LHS[i, :]= 0
                #     LHS[i][i] = 1
                #     RHS[i] = 0
                # if self.mesh.vertices[i].which_boundary == 'x':
                #     LHS[i, :]= 0
                #     LHS[i][i] = 1
                #     RHS[i] = 0
        return LHS, RHS
    def Solve(self):
        LHS, RHS = self.Assembly()
        solution = sparse.linalg.spsolve(sparse.csr_matrix(LHS), RHS)##taking advantage of the sparsity of the matrix
        return solution
    def PlotSolution(self):
        solution = self.Solve()
        soln = []
        for index in range(len(solution)):
            if self.mesh.vertices[index].wd == 'val':
                soln.append(solution[index])
        soln = np.array(soln)
        print(f"{np.round(np.max(soln),6)}")
        n = len(soln)
        soln = soln.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
        ## inspired by https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html#sphx-glr-gallery-mplot3d-trisurf3d-2-py
        x = y = np.linspace(-1, 1, int(1/self.mesh.h) +1 )
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
        plt.title(fr"Solution to $\nabla^4 u = 1$, with homogenous BCs, computed on a mesh $h ={self.mesh.h} $")
        #plt.savefig(r"C:\Users\borys\OneDrive\Dokumenty\cplusplus\MMATH_project\figures\biharm32.jpg", dpi = 500)
    def AssemblyEigen(self):
        ElementMatrix = self.ElementIntegrationLHS()
        ElementRHS = self.ElementIntegrationEigenRHS()
        ConnectivityMatrix = self.mesh.ConnectivityMatrix()
        matrixSize = int(np.size(self.mesh.vertices))
        LHS = np.zeros((matrixSize, matrixSize))
        RHS = np.zeros((matrixSize, matrixSize))
        Idid = np.ones(matrixSize)
        ## loop as in https://eprints.maths.manchester.ac.uk/894/2/0-19-852868-X.pdf p.26
        for k in range(len(self.mesh.triangles)):
            for j in range(12):
                fix_j = 1
                if any([j == 9, j==10, j==11]):
                    fix_j = Idid[int(ConnectivityMatrix[k][j])]
                for i in range(12):
                    fix_i = 1
                    if any([i == 9, i==10, i==11]):
                        fix_i = Idid[int(ConnectivityMatrix[k][i])]
                    #  if check_elements_in_arrays([int(ConnectivityMatrix[k][j]), int(ConnectivityMatrix[k][i])], edges[0], edges[1], edges[2]):
                    LHS[int(ConnectivityMatrix[k][j])][int(ConnectivityMatrix[k][i])]+=fix_i * fix_j * ElementMatrix[k][j][i]
                    RHS[int(ConnectivityMatrix[k][j])][int(ConnectivityMatrix[k][i])]+=fix_i * fix_j * ElementRHS[k][j][i]
            Idid[self.mesh.triangles[k].VertexNumbers[9:]] = -1 ##as in the mysterious matlab file
        ## at this point, the galerkin matrix is of size n by n where n is the total number of vertices
        ## we have enforce the boundary conditions on the system
        for i in range(len(self.mesh.vertices)):
            if self.mesh.vertices[i].boundary:
                if self.mesh.vertices[i].wd == 'val':
                    LHS[i, :]= 0
                    LHS[i][i] = 1
                    RHS[i,:] = 0
                    RHS[i][i] = 1
                ### a bit tricky part
                # we set the directional boundary conditions to 0 on the counterpart boundary
                # ie u_x = 0 on y = 0 y =1 
                # u_y = 0 on x = 0 x = 1
                if self.mesh.vertices[i].wd == 'norm':
                    LHS[i, :]= 0
                    LHS[i][i] = 1
                    RHS[i,:] = 0
                    RHS[i][i] = 1
        return LHS, RHS
    def SolveEigen(self, sigma = 80):
        LHS, RHS = self.AssemblyEigen()
        eigs, vecs = sparse.linalg.eigs(sparse.csr_matrix(LHS), 6, sparse.csr_matrix(RHS), sigma = sigma)
        return eigs, vecs
    def PlotEigen(self, sigma = 80):
        eigs, vecs = self.SolveEigen(sigma)
        vecs = vecs[:,0].real
        soln = []
        for index in range(len(vecs)):
            if self.mesh.vertices[index].wd == 'val':
                soln.append(vecs[index])
        soln = np.array(soln)


        n = len(soln)
        soln = soln.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
        ## inspired by https://matplotlib.org/stable/gallery/mplot3d/trisurf3d_2.html#sphx-glr-gallery-mplot3d-trisurf3d-2-py
        x = y = np.linspace(-1, 1, int(1/self.mesh.h) +1 )
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
        plt.title(fr" $\nabla^4 u = \lambda u$ solution corresponding to numerical eigenvalue $\lambda = {np.round(eigs[0].real, 4)}$")
        plt.show()
    def PlotContour(self, sigma = 80):
        eigs, vecs = self.SolveEigen(sigma)
        print(eigs)
        vecs = vecs[:,0].real
        soln = []
        for index in range(len(vecs)):
            if self.mesh.vertices[index].wd == 'val':
                soln.append(vecs[index])
        soln = np.array(soln)


        n = len(soln)
        soln = soln.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
        x = y = np.linspace(-1, 1, int(1/self.mesh.h) +1 )
        x, y = np.meshgrid(x, y)
        plt.contour(x, y, soln)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(fr" $\nabla^4 u = \lambda u$ solution corresponding to numerical eigenvalue $\lambda = {np.round(eigs[0].real, 4)}$")
        # plt.colorbar()
        plt.show()
    def Plot4eigenfunctions(self):
        expected_eigenvalues = [80, 336, 731, 1082]
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(fr"Eigenfunctions solving $\nabla^4 u = \lambda u$ with indicated eigenvalues $\lambda$", fontsize=16)  # Add global title
        pairs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i, pair in enumerate(pairs):
            eigs, vecs = self.SolveEigen(expected_eigenvalues[i])
            vecs = vecs[:, 0].real
            soln = []
            for index in range(len(vecs)):
                if self.mesh.vertices[index].wd == 'val':
                    soln.append(vecs[index])
            soln = np.array(soln)
            n = len(soln)
            soln = soln.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
            x = y = np.linspace(-1, 1, int(1/self.mesh.h) + 1)
            x, y = np.meshgrid(x, y)
            z = soln.flatten()
            tri = mtri.Triangulation(x.flatten(), y.flatten())
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.plot_trisurf(tri, z, cmap=plt.cm.Spectral_r)
            ax.set_xlabel(f'$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel(f'$z$')
            # Add bounding box
            ax.text2D(0.05, 0.95, fr"$\lambda = {np.round(eigs[0].real, 4)}$", transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.tight_layout()
        plt.savefig("eigenfunctions.png", dpi = 600)
