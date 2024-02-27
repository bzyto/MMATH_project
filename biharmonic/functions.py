import numpy as np
import matplotlib.pyplot as plt
from Elements import *
from mesh import *
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
            triangles.append(Triangle([v1.coordinates, v4.coordinates, v2.coordinates], [v1.global_number, v1_x.global_number, v1_y.global_number, v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v4, v2)), midpoints_dict.get(get_tuple(v1, v2)), 
                                                                                          midpoints_dict.get(get_tuple(v1, v4))], loopcounter,
                                                                                        edg = [[v1.global_number, v1_x.global_number, v1_y.global_number, v2.global_number, v2_x.global_number, v2_y.global_number, midpoints_dict.get(get_tuple(v1, v2))],
                                                                                               [v1.global_number, v1_x.global_number, v1_y.global_number, v4.global_number, v4_x.global_number, v4_y.global_number,midpoints_dict.get(get_tuple(v1, v4))],
                                                                                               [v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v4, v2))]])) ##lower triangle
            triangles.append(Triangle([v3.coordinates, v2.coordinates, v4.coordinates], [v3.global_number, v3_x.global_number, v3_y.global_number, v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v2, v4)), midpoints_dict.get(get_tuple(v3, v4)),
                                                                                          midpoints_dict.get(get_tuple(v3, v2))], loopcounter+1,
                                                                                          edg = 
                                                                                              [[v3.global_number, v3_x.global_number, v3_y.global_number, v2.global_number, v2_x.global_number, v2_y.global_number,midpoints_dict.get(get_tuple(v3, v2))],
                                                                                               [v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v2, v4))],
                                                                                          [v3.global_number, v3_x.global_number, v3_y.global_number,v4.global_number, v4_x.global_number, v4_y.global_number,midpoints_dict.get(get_tuple(v3, v4))]
                                                                                              ])) ##upper triangle
            loopcounter+=2
    all_but_coms = []
    vertices = vertices.flatten()
    x_vertices = x_vertices.flatten()
    y_vertices = y_vertices.flatten()
    for i in range(len(vertices)):
        all_but_coms.append(vertices[i])
        all_but_coms.append(x_vertices[i])
        all_but_coms.append(y_vertices[i])
    nomids = len(all_but_coms)
    for mid in midpoints:
        all_but_coms.append(mid)
    all_but_coms.sort(key = lambda x: x.global_number)
    triangles.sort(key = lambda x: x.GlobalNumber)
    return Mesh(all_but_coms, triangles, h, nomids)
def exact_biharm(x = 0.5, y = 0.5):
    max_i = 100
    max_j = 100
    val = 0
    for j in range(1,max_j):
        for k in range(1,max_i):
            if (j%2 == 1 and k%2 == 1):
                val+=np.sin(j*np.pi*x)*np.sin(k*np.pi*y) * 1/((j*k)*(k**2+j**2)**2)
    return val*(16/np.pi**6)
def generateMesh11square(h = 0.2):
    x = y = np.linspace(-1, 1, int(2/h)+1)
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
            if x_grid[i][j] == -1 or x_grid[i][j] == 1 or y_grid[i][j] == -1 or y_grid[i][j] == 1:
                boundary = True
            if i%2 == 0 and j%2==0:
                vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter, boundary, wd = 'val'))
                x_vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter+1, boundary, wb, wd = 'x'))
                y_vertices.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter+2, boundary, wb, wd = 'y'))
                loopcounter+=3
            else:
                midpoints.append(Vertex([x_grid[i][j], y_grid[i][j]], loopcounter, boundary, wb, wd = 'norm'))
                midpoints_dict[(np.round(x_grid[i][j], 10), np.round(y_grid[i][j], 10))]= loopcounter
                loopcounter+=1
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
            triangles.append(Triangle([v1.coordinates, v4.coordinates, v2.coordinates], [v1.global_number, v1_x.global_number, v1_y.global_number, v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v4, v2)), midpoints_dict.get(get_tuple(v1, v2)), 
                                                                                          midpoints_dict.get(get_tuple(v1, v4))], loopcounter,
                                                                                        edg = [[v1.global_number, v1_x.global_number, v1_y.global_number, v2.global_number, v2_x.global_number, v2_y.global_number, midpoints_dict.get(get_tuple(v1, v2))],
                                                                                               [v1.global_number, v1_x.global_number, v1_y.global_number, v4.global_number, v4_x.global_number, v4_y.global_number,midpoints_dict.get(get_tuple(v1, v4))],
                                                                                               [v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v4, v2))]])) ##lower triangle
            triangles.append(Triangle([v3.coordinates, v2.coordinates, v4.coordinates], [v3.global_number, v3_x.global_number, v3_y.global_number, v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v2, v4)), midpoints_dict.get(get_tuple(v3, v4)),
                                                                                          midpoints_dict.get(get_tuple(v3, v2))], loopcounter+1,
                                                                                          edg = 
                                                                                              [[v3.global_number, v3_x.global_number, v3_y.global_number, v2.global_number, v2_x.global_number, v2_y.global_number,midpoints_dict.get(get_tuple(v3, v2))],
                                                                                               [v2.global_number, v2_x.global_number, v2_y.global_number,
                                                                                          v4.global_number, v4_x.global_number, v4_y.global_number,
                                                                                          midpoints_dict.get(get_tuple(v2, v4))],
                                                                                          [v3.global_number, v3_x.global_number, v3_y.global_number,v4.global_number, v4_x.global_number, v4_y.global_number,midpoints_dict.get(get_tuple(v3, v4))]
                                                                                              ])) ##upper triangle
            loopcounter+=2
    all_but_coms = []
    vertices = vertices.flatten()
    x_vertices = x_vertices.flatten()
    y_vertices = y_vertices.flatten()
    for i in range(len(vertices)):
        all_but_coms.append(vertices[i])
        all_but_coms.append(x_vertices[i])
        all_but_coms.append(y_vertices[i])
    nomids = len(all_but_coms)
    for mid in midpoints:
        all_but_coms.append(mid)
    all_but_coms.sort(key = lambda x: x.global_number)
    triangles.sort(key = lambda x: x.GlobalNumber)
    return Mesh(all_but_coms, triangles, h, nomids)