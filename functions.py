import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import cv2
import skimage.filters
import scipy.misc
import skimage.color
import scipy.sparse
import scipy.ndimage.interpolation
import scipy.spatial
import scipy.interpolate
from skimage.draw import polygon
from scipy.interpolate import interp2d

#Reads an image with given path as grayscale
def readGrayScale(path):
    image = skio.imread(path)
    image = sk.img_as_float(image)
    return sk.color.rgb2gray(image)

def read(path):
    return skio.imread(path)

#Reads an image with given path as array of color channels
def readColor(path):
    image = skio.imread(path)
    image_full = sk.img_as_float(image)
    return image_full

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

def pltShow(image):
    plt.imshow(image)
    plt.show()

#Saves image to given path
def saveImage(path, image):
    skio.imsave(path, image)

def showTri(image, points, tri):
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()

#Shows a Delaunay triagulation over an image
def showTriSave(image, points, tri, name):
    figure = plt.figure()
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()
    figure.savefig(name)

#Return homogenous matrix containing points
def homogenous(points):
    return np.row_stack((points.transpose(), (1, 1, 1)))


#Takes a homogenous matrix and recovers the points
def homo_to_points(homo_matrix):
    values = homo_matrix[:2]
    print(values.transpose())

def np_clip(img):
    return np.clip(img, 0, 1)

#Rescaels image between 0 and 1
def rescale(img):
    return (img - img.min()) / (img.max() - img.min())


def L2Norm(image):
    np.linalg.norm(image)

def square_root(matrix):
    return np.sqrt(matrix)

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
    return cv2.filter2D(image, -1, kernel)


#Gaussian stack of IMAGE, with DIM_FACTOR the size of the Gaussian kernel
def GaussianStack(image, dim_factor, sigma, stack_depth):
    stack = []
    for i in range(stack_depth + 1):
        image = lowPass(image, dim_factor, sigma)
        sigma *= 2
        stack.append(image)
    return stack

#Laplacian stack given a Gaussian stack
def LaplacianStack(image, stack):
    lap_stack = []
    lap_stack.append(rescale(image - stack[0]))
    for i in range(1, len(stack) - 1):
        lap = stack[i] - stack[i+1]
        lap = rescale(lap)
        lap_stack.append(lap)
    return lap_stack

"""
This functions works to redefine low passing, only this time with a mask.
"""
def lowPassMask(image, size, sigma, mask):
    mask_G = lowPass(mask, size, sigma)
    output = lowPass(image*mask, size, sigma)
    #showImage(output)
    output = rescale(output / mask_G)
    #showImage(output)
    return output
"""
This method is based specifically on the matlab version of the paper.
One thing it does is increase the size of the Gaussian kernel at each level
Likewise, I try to match the implementation as much as possible.
This version also takes in a mask but haven't added that part yet.
Read section 'Using a Mask' in the paper for why this is important.
"""
def LaplacianStackAlt(image, mask, stack_depth):
    stack = []
    stack.append(image)
    for i in range(1, stack_depth):
        sigma = 2 ** i
        #stack.append(lowPassMask(image, sigma*5, sigma, mask))
        stack.append(lowPass(image, sigma*5, sigma))

    for i in range(len(stack)-1):
        stack[i] = rescale(stack[i] - stack[i+1])
        #stack[i] = rescale((stack[i] - stack[i+1])*mask)
    return stack

#Aggregates all images in STACK
def sumStack(stack):
    final_image = stack[0]
    for i in range(1, len(stack)):
        final_image += stack[i]
    return rescale(final_image)

#Warps IMAGE with SOURCE_POINTS to TARGET_POINTS with TRI
def warp(image, source_points, target_points, tri):
    imh, imw = image.shape
    out_image = np.zeros((imh, imw))
    xs = np.arange(imw)
    ys = np.arange(imh)
    interpFN = interpFunc(xs, ys, image)

    for triangle_indices in tri.simplices:

        source_triangle = source_points[triangle_indices]
        target_triangle = target_points[triangle_indices]
        A = computeAffine(source_triangle, target_triangle)
        A_inverse = np.linalg.inv(A)

        tri_rows = target_triangle.transpose()[1]
        tri_cols = target_triangle.transpose()[0]

        row_coordinates, col_coordinates = polygon(tri_rows, tri_cols)

        for x, y in zip(col_coordinates, row_coordinates):
            #point inside target triangle mesh
            point_in_target = np.array((x, y, 1))

            #point inside source image
            point_on_source = np.dot(A_inverse, point_in_target)

            x_source = point_on_source[0]
            y_source = point_on_source[1]

            source_value = interpFN(x_source, y_source)
            try:
                out_image[y, x] = source_value
            except IndexError:
                continue

    return out_image

"""
def getInvWarpMat(triangleIndices, srcKeypoints, avgKeypoints):
    # Initialize arrays for holding x and y indices of triangle vertices
    srcImg_x = np.zeros(3)
    srcImg_y = np.zeros(3)
    avgImg_x = np.zeros(3)
    avgImg_y = np.zeros(3)

    # grab correct indices
    index = 0
    for j in triangleIndices:
        avgImg_x[index] = avgKeypoints[j][0]  # X indices of triangle vertices (index into 1 because points are (column, row))
        avgImg_y[index] = avgKeypoints[j][1]  # Y indices of triangle vertices (index into 0 because points are (column, row))
        srcImg_x[index] = srcKeypoints[j][0]  # X indices of triangle vertices (index into 1 because points are (column, row))
        srcImg_y[index] = srcKeypoints[j][1]  # Y indices of triangle vertices (index into 0 because points are (column, row))
        index += 1

    # Create matrix for source image
    srcCol_0 = np.array([srcImg_x[0], srcImg_y[0], 1])  # One column is in the form (x1, y1, 1)
    srcCol_1 = np.array([srcImg_x[1], srcImg_y[1], 1])  # (x2, y2, 1)
    srcCol_2 = np.array([srcImg_x[2], srcImg_y[2], 1])  # (x3, y3, 1)
    srcImgMatrix = np.column_stack((srcCol_0, srcCol_1, srcCol_2))  # [[x1, x2, x3],
    # [y1, y2, y3],
    # [ 1,  1,  1]]

    # Create matrix for midpoint image
    avgCol_0 = np.array([avgImg_x[0], avgImg_y[0], 1])  # One column is in the form (x1, y1, 1)
    avgCol_1 = np.array([avgImg_x[1], avgImg_y[1], 1])  # (x2, y2, 1)
    avgCol_2 = np.array([avgImg_x[2], avgImg_y[2], 1])  # (x3, y3, 1)
    avgImgMatrix = np.column_stack((avgCol_0, avgCol_1, avgCol_2))  # [[x1, x2, x3],
    # [y1, y2, y3],
    # [ 1,  1,  1]]

    # Compute T transformation matrix
    # M = Midway Matrix
    # M = TA -------> T = MA^(-1)
    transformMat = scipy.matmul(avgImgMatrix, scipy.linalg.inv(srcImgMatrix))

    # Return T^(-1) so that we can use inverse transformation
    return scipy.linalg.inv(transformMat)

def morph(imgA, imgB, srcKeypoints, targetKeypoints, warp_frac, dissolve_frac, IS_GRAY=False):
    midway = np.zeros(imgB.shape)

    avgKeypoints = []
    for i in range(len(srcKeypoints)):
        avg_x = srcKeypoints[i][0]*(1 - warp_frac) + targetKeypoints[i][0] * warp_frac
        avg_y = srcKeypoints[i][1]*(1 - warp_frac) + targetKeypoints[i][1] * warp_frac
        avgPt = (avg_x, avg_y)
        avgKeypoints.append(avgPt)
    avgKeypoints = np.array(avgKeypoints)

    midTriangulation = scipy.spatial.Delaunay(avgKeypoints)

    height = np.arange(len(imgA))
    width = np.arange(len(imgA[0]))
    if IS_GRAY:
        interpFuncGreyA = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgA)
        interpFuncGreyB = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgB)
    else:
        interpFuncA_R = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgA[:, :, 0])
        interpFuncA_G = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgA[:, :, 1])
        interpFuncA_B = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgA[:, :, 2])

        interpFuncB_R = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgB[:, :, 0])
        interpFuncB_G = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgB[:, :, 1])
        interpFuncB_B = scipy.interpolate.interp2d(np.arange(len(width)), np.arange(len(height)), imgB[:, :, 2])

    # Iterate through each triangle
    for i in range(len(midTriangulation.simplices)):
        midTriangleIndices = midTriangulation.simplices[i]  # Triangle of midway image
        # Initialize arrays to hold x,y coordinates of triangle vertices
        avgImg_x = np.zeros(3)
        avgImg_y = np.zeros(3)

        #     Iterate through the 3 vertices in the triangle
        #     j is a number representing the index of the point in the keypoints
        index = 0
        for j in midTriangleIndices:
            avgImg_x[index] = avgKeypoints[j][1]  # X indices of triangle vertices (index into 1 because points are (column, row))
            avgImg_y[index] = avgKeypoints[j][0]  # Y indices of triangle vertices (index into 0 because points are (column, row))
            index += 1

        midTriangle = avgKeypoints[midTriangleIndices]

        # Mid = M
        # M = TA -> T = (MA^(-1))
        AtoMidInvWarpMat = getInvWarpMat(midTriangleIndices, srcKeypoints, avgKeypoints)

        # M = TB -> T = (MB^(-1))
        BtoMidInvWarpMat = getInvWarpMat(midTriangleIndices, targetKeypoints, avgKeypoints)

        # polygon(r, c, shape=None) -> rr, cc
        # r = row coordinates of vertices in polygon
        # c = column coordinates of vertices in polygon
        # Returns row coords and col coords of image
        triangleRows, triangleCols = skimage.draw.polygon(avgImg_x, avgImg_y)

        # Iterate through points of the form (x,y)
        for x, y in zip(triangleCols, triangleRows):
            # Create array for point of final img
            midImgPt = np.array([x, y, 1])

            # Inverse warp on pt in midpoint gets us point in A
            APt = np.dot(AtoMidInvWarpMat, midImgPt)
            AX, AY = APt[0], APt[1]

            # Inverse warp on pt in midpoint gets us point in B
            BPt = np.dot(BtoMidInvWarpMat, midImgPt)
            BX, BY = BPt[0], BPt[1]

            if IS_GRAY:
                #Interpolate on both points to get respective values
                AinterpVal = interpFuncGreyA(AX, AY)
                BinterpVal = interpFuncGreyB(BX, BY)

                #Average value and set midway point
                midwayVal = AinterpVal*(1 - dissolve_frac) + BinterpVal * dissolve_frac
                midway[y, x] = midwayVal

            else:
                A_redInterpVal = interpFuncA_R(AX, AY)
                A_greenInterpVal = interpFuncA_G(AX, AY)
                A_blueInterpVal = interpFuncA_B(AX, AY)
                A_val = np.array([A_redInterpVal, A_greenInterpVal, A_blueInterpVal])

                B_redInterpVal = interpFuncB_R(BX, BY)
                B_greenInterpVal = interpFuncB_G(BX, BY)
                B_blueInterpVal = interpFuncB_B(BX, BY)
                B_val = np.array([B_redInterpVal, B_greenInterpVal, B_blueInterpVal])

                midwayRed = A_redInterpVal*(1 - dissolve_frac) + B_redInterpVal * dissolve_frac
                midwayGreen = A_greenInterpVal*(1 - dissolve_frac) + B_greenInterpVal * dissolve_frac
                midwayBlue = A_blueInterpVal*(1 - dissolve_frac) + B_blueInterpVal * dissolve_frac
                midway[y, x, 0] = midwayRed
                midway[y, x, 1] = midwayGreen
                midway[y, x, 2] = midwayBlue

    midway = (midway - np.min(midway) ) / (np.max(midway) - np.min(midway))
    # skio.imshow(midway)
    # skio.show()
    return midway
"""
