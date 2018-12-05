import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.util
import skimage.io as skio
import skimage.color
import cv2
import scipy.sparse
import scipy.ndimage.interpolation
import math
import skimage.transform as sktr
from imutils import face_utils
import imutils
import argparse
import dlib
from functions import *

def getFacialLandmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    plt.figure()
    triangulation = None
    shape = None
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # # convert dlib's rectangle to a OpenCV-style bounding box
        # # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # show the face number
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            plt.scatter(x, y, s=10)


        corners = [(0,0), (image.shape[1],0), (0, image.shape[0]), (image.shape[1], image.shape[0])]
        
        for corner in corners:
            shape = np.vstack((shape, corner))
        triangulation = scipy.spatial.Delaunay(shape)
    
    return shape, triangulation

def getGaussianStacks(inputIm, exampleIm):
    gStackInput = GaussianStack(inputIm, 45, 2, 3)
    gStackExample = GaussianStack(exampleIm, 45, 2, 3)
    
    return gStackInput, gStackExample

def getLaplacianStacks(gStackInput, gStackExample, inputIm, exampleIm):
    lStackInput = LaplacianStack(inputIm, gStackInput)
    lStackExample = LaplacianStack(exampleIm, gStackExample)
    # for im in lStackExample:
    #     showImage(im)
    return lStackInput, lStackExample

def getResidualStack(img, imgStack):
    residualStack = []
    for g in imgStack:
        residualStack.append(cv2.convolve(img, g))
    return residualStack

def getLocalEnergyStack(gStack, lStack):
    energyStack = []
    for i in range(len(lStack)):
        laplacian = lStack[i]
        laplacian_sq = np.square(laplacian)
        energyStack.append(lowPass(laplacian_sq, 45, 2 ** (i+1)))
    return energyStack

def warpEnergyStack(eStack, inputShape, inputTriangulation, exampleShape, exampleTriangulation):
    warpedStack = []
    for elem in eStack:
        #WARP EVERY TRIANGLE FROM EXAMPLE TRIANGULATION TO INPUT TRIANGULATION HERE
    return warpedStack

jose = skio.imread('jose.jpg')
george = skio.imread('george.jpg')
gStackJose, gStackGeorge = getGaussianStacks(jose, george)
lStackJose, lStackGeorge = getLaplacianStacks(gStackJose, gStackGeorge, jose, george)
getLocalEnergyStack(gStackJose, lStackJose)
inputShape, inputTriangulation = getFacialLandmarks(jose)
exampleShape, exampleTriangulation = getFacialLandmarks(george)
print(inputShape)



