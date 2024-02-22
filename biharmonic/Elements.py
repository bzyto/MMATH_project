import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

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
class Triangle:
    def __init__(self, vertices, VertexNumbers =0, GlobalNumber = 0, edg=[]):
        self.x_0 = np.array(vertices[0])
        self.x_1 = np.array(vertices[1])
        self.x_2 = np.array(vertices[2])
        self.vertices = vertices
        self.GlobalNumber = GlobalNumber
        self.VertexNumbers = VertexNumbers
        self.Edges = edg
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
            np.concatenate([-self.OuterUnitNormal()[0][0]*phi_x(1/2 * (self.x_0+self.x_1))-
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
            np.concatenate([np.zeros(10), -self.OuterUnitNormal()[1][0]*phi_x(1/2 * (self.x_1+self.x_2))-
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
        pairs = [[0,25], [1,26], [2, 27], [3, 10], [4, 11], [5, 12], [13, 22], [14, 23], [15, 24], [21], [29], [9]]  # enforcing the correct conditions on shape functions
        sols = []
        LU = lu_factor(LHS)  # Perform LU decomposition of LHS matrix

        for i in range(12):
            RHS = np.zeros(30)
            for index in pairs[i]:
                RHS[index] = 1
            sols.append(lu_solve(LU, RHS))  # Solve the system using LU decomposition
        return sols