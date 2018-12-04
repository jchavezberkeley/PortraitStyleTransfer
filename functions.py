import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import cv2
from skimage.draw import polygon
from scipy.interpolate import interp2d

#Reads an image with given path as grayscale
def readGrayScale(path):
    image = skio.imread(path)
    image = sk.img_as_float(image)
    return sk.color.rgb2gray(image)


#Reads an image with given path as array of color channels
def readColors(path):
    image = skio.imread(path)
    image_full = sk.img_as_float(image)
    image_colors = image_full.transpose(2, 0, 1)
    return image_colors


#Shows points on a given face image
def showPoints(image, points):
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()


def showImage(image):
    skio.imshow(image)
    skio.show()


#Saves image to given path
def saveImage(path, image):
    skio.imsave(path, image)


#Shows a Delaunay triagulation over an image
def showTri(image, points, tri):
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()

#Return homogenous matrix containing points
def homogenous(points):
    return np.row_stack((points.transpose(), (1, 1, 1)))


#Takes a homogenous matrix and recovers the points
def homo_to_points(homo_matrix):
    values = homo_matrix[:2]
    print(values.transpose())


"""
Used to recover the affine transformation given two sets of points
tri_points2 = np.dot(tri_points1, R) + t
where R is the rotation portion.

Solution found at https://stackoverflow.com/questions/27546081/
determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi
"""
def computeAffine(tri_points1, tri_points2):
    #A is going to be the matrix with tri_points1
    source_matrix = homogenous(tri_points1)
    target_matrix = homogenous(tri_points2)
    T = np.matmul(target_matrix, np.linalg.inv(source_matrix))
    return T


#Linear interpolations
def interp(fraction, item1, item2):
    return ((1-fraction) * item1) + (fraction * item2)

#Returns an interp2d function
def interpFunc(xs, ys, image):
    return interp2d(xs, ys, image)

#Low pass of a given IMAGE with kernel SIZE and SIGMA
def lowPass(image, size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = np.multiply(kernel, kernel.transpose())
    return np.clip(cv2.filter2D(image, -1, kernel), 0, 255)

#Gaussian stack of IMAGE, with DIM_FACTOR the size of the Gaussian kernel
def GaussianStack(image, dim_factor, sigma, stack_depth):
    stack = []
    for i in range(stack_depth):
        image = lowPass(image, dim_factor, sigma)
        sigma *= 2
        stack.append(image)
    return stack

#Laplacian stack given a Gaussian stack
def LaplacianStack(image, stack):
    lap_stack = []
    lap_stack.append(image - stack[0])
    for i in range(1, len(stack) - 1):
        lap = stack[i] - stack[i+1]
        lap_stack.append(lap)
    last_level = stack[len(stack) - 1]
    last_kernel = lowPass(image, 45, len(stack) ** 2)  
    last_level = last_level - last_kernel
    lap_stack.append(last_level)
    return lap_stack
