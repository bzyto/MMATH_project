import cv2
from scipy import ndimage
import numpy as np
from PoissonLinearApprox import Triangle, Mesh, Vertex, PoissonSolver
def get_boundary(binary_image):
    # Erode the binary image
    eroded_image = ndimage.binary_erosion(binary_image)

    # Subtract the eroded image from the original image to get the boundary
    boundary_image = binary_image - eroded_image

    return boundary_image

# Use the function
def lower_resolution(input_path, output_path, scale_percent):
    # Load the original image
    original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the new dimensions of the image
    original_width = int(original_image.shape[1])
    original_height = int(original_image.shape[0])
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(original_image, (new_width, new_height))

    # Convert the resized image to binary format
    _, binary_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite(output_path, binary_image)

    print(f"Binary image saved to {output_path}")

    # Return the binary image
    return binary_image

# Replace 'your_image_path.jpg' with the actual path to your image file

# Set the scale percentage (e.g., 50% for half the resolution)
def remove_boundary_ones(boundary_image):
    # Set the first and last row to 0
    boundary_image[0, :] = 0
    boundary_image[-1, :] = 0

    # Set the first and last column to 0
    boundary_image[:, 0] = 0
    boundary_image[:, -1] = 0

    return boundary_image
from scipy import ndimage

def remove_isolated_ones(binary_image):
    # Define the kernel that will be used for convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Count the number of neighbors for each cell
    neighbors_count = ndimage.convolve(binary_image, kernel, mode='constant', cval=0)

    # Set the cells with no neighbors to 0
    binary_image[(binary_image == 1) & (neighbors_count == 0)] = 0

    return binary_image
def binary_matrix_to_image(binary_matrix):
    # Convert the binary matrix to an 8-bit image
    image = (binary_matrix * 255).astype(np.uint8)

    # Write the image to a file
    cv2.imshow('dupa', image)
    cv2.waitKey(0)
from scipy import ndimage

def fill_inside_boundary(binary_image):
    # Fill the holes in the binary image
    filled_image = ndimage.binary_fill_holes(binary_image)

    return filled_image
scale_percent = 30
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
plt.rcParams['text.usetex'] = True #for nice plots

def generate_mesh(binary_matrix):
    # Find the indices of the 1's in the binary matrix
    y, x = np.where(binary_matrix == 1)

    # Create a triangulation of these points
    triangulation = tri.Triangulation(x, y)

    # Calculate the lengths of the sides of each triangle
    x = triangulation.x[triangulation.triangles]
    y = triangulation.y[triangulation.triangles]
    a = np.sqrt((x[:, 0] - x[:, 1])**2 + (y[:, 0] - y[:, 1])**2)
    b = np.sqrt((x[:, 1] - x[:, 2])**2 + (y[:, 1] - y[:, 2])**2)
    c = np.sqrt((x[:, 2] - x[:, 0])**2 + (y[:, 2] - y[:, 0])**2)

    # Sort the sides so that a <= b <= c
    a, b, c = np.sort(np.stack([a, b, c], axis=-1), axis=-1).T

    # Check if the Pythagorean theorem holds and if b/c is less than np.sqrt(3)/2
    mask = np.isclose(a**2 + b**2, c**2) & (b/c < np.sqrt(3)/2)

    # Remove the non-right triangles
    triangulation.set_mask(~mask)

    return triangulation
def find_boundary_vertices(triangulation):
    # Get the edges
    edges = triangulation.edges

    # Flatten the edges array and count the number of occurrences of each point
    point_counts = np.bincount(edges.ravel())
    # Find the points that appear only once, which are the boundary points
    boundary_points = np.where(point_counts==5)[0]

    return boundary_points

def generate_vertices(triangulation):
    bound = find_boundary_vertices(triangulation)
    vertices =[]
    for k in range(len(triangulation.x)):
        if k in bound:
            b = True
        else:
            b= False
        vertices.append(Vertex([triangulation.x[k], triangulation.y[k]], k, b))
    return vertices
def generate_mytriangles(triangulation):
    trigs = []
    for index, triangle in enumerate(triangulation.triangles):
        point1 = triangulation.x[triangle[0]], triangulation.y[triangle[0]]
        point2 = triangulation.x[triangle[1]], triangulation.y[triangle[1]]
        point3 = triangulation.x[triangle[2]], triangulation.y[triangle[2]]
        trigs.append(Triangle([point1, point2, point3], triangle, index))
    return trigs
