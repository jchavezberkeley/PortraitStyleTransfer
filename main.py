import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
#import skimage.util as skutil
import skimage.io as skio
#import skimage.color as skcolor
import cv2
import scipy.sparse
import scipy.ndimage.interpolation
import math
import skimage.transform as sktr
from imutils import face_utils
import imutils
import argparse
import dlib
from skimage.draw import polygon
from scipy.interpolate import interp2d
from functions import *

def getFacialLandmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    #plt.figure()
    triangulation = None
    shape = None
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(image, rect)
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
        #for (x, y) in shape:
            #plt.scatter(x, y, s=10)


        corners = [(0,0), (image.shape[1],0), (0, image.shape[0]), (image.shape[1], image.shape[0])]

        for corner in corners:
            shape = np.vstack((shape, corner))
        triangulation = scipy.spatial.Delaunay(shape)

    return shape, triangulation

def getGaussianStacks(inputIm, exampleIm, stack_depth):
    gStackInput = GaussianStack(inputIm, 45, 2, stack_depth)
    gStackExample = GaussianStack(exampleIm, 45, 2, stack_depth)
    return gStackInput, gStackExample

def getLaplacianStacks(gStackInput, gStackExample, inputIm, exampleIm):
    lStackInput = LaplacianStack(inputIm, gStackInput)
    lStackExample = LaplacianStack(exampleIm, gStackExample)
    return lStackInput, lStackExample

def getResidualStack(img, imgStack):
    residualStack = []
    for g in imgStack:
        residualStack.append(cv2.convolve(img, g))
    return residualStack

def getResidual(image, stack_depth):
    return lowPass(image, 45, 2**stack_depth)

def getLocalEnergyStack(lStack):
    energyStack = []
    for i in range(len(lStack)):
        laplacian = lStack[i]
        laplacian_squared = np.square(laplacian)
        energy = lowPass(laplacian_squared, 45, 2 ** (i+1))
        energyStack.append(energy)
    return energyStack

def warpEnergyStack(inputIm, eStack, inputShape, inputTri, exampleShape, exampleTri):
    warpedStack = []
    for elem in eStack:
        #WARP EVERY TRIANGLE FROM EXAMPLE TRIANGULATION TO INPUT TRIANGULATION HERE
        #warped = morph(inputIm, elem, inputShape, exampleShape, 0, 1, IS_GRAY=False)
        warped = warp(elem, exampleShape, inputShape, inputTri)
        warpedStack.append(warped)
    return warpedStack

def robustTransfer(inputLapStack, warpedStack, exampleEnergyStack):
    newGainStack = []
    epsilon = 0.01 ** 2
    for i in range(len(inputLapStack)):
        division = (warpedStack[i] / (exampleEnergyStack[i] + epsilon))
        gain = square_root(division)
        newLayer = inputLapStack[i] * gain
        newGainStack.append(newLayer)
    return newGainStack

jose = read('jose.jpg')
george = read('george.jpg')
jose_gray = readGrayScale('jose.jpg')
george_gray = readGrayScale('george.jpg')

inputShape, inputTri = getFacialLandmarks(jose)
exampleShape, exampleTri = getFacialLandmarks(george)
#morph_example = warp(george_gray, exampleShape, inputShape, inputTri)

stack_depth = 3
gStackJose, gStackGeorge = getGaussianStacks(jose_gray, george_gray, stack_depth)
lStackJose, lStackGeorge = getLaplacianStacks(gStackJose, gStackGeorge, jose_gray, george_gray)

jose_residual = getResidual(jose_gray, stack_depth)
george_residual = getResidual(george_gray, stack_depth)

joseEStack = getLocalEnergyStack(lStackJose)
georgeEStack = getLocalEnergyStack(lStackGeorge)

warpedStack = warpEnergyStack(jose_gray, georgeEStack, inputShape, inputTri, exampleShape, exampleTri)
for w in warpedStack:
    showImage(w)
gainStack = robustTransfer(lStackJose, warpedStack, georgeEStack)
for g in gainStack:
    showImage(g)

#showTri(morph_example, inputShape, inputTri)
#showTri(jose, inputShape, inputTri)
#showTri(george, exampleShape, exampleTri)
#for im in warpedStack:
    #showImage(im)
