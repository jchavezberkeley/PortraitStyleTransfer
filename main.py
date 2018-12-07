import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import cv2
#import scipy.sparse
#import scipy.ndimage.interpolation
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

def getLaplacianStacks(inputIm, exampleIm, input_mask, example_mask, stack_depth):
    lStackInput = LaplacianStackAlt(inputIm, input_mask, stack_depth)
    lStackExample = LaplacianStackAlt(exampleIm, example_mask, stack_depth)
    return lStackInput, lStackExample

def getResidualStack(img, imgStack):
    residualStack = []
    for g in imgStack:
        residualStack.append(cv2.convolve(img, g))
    return residualStack

def getResidual(image, mask, stack_depth):
    return lowPass(image, 5*(2**stack_depth), 2**stack_depth)

def getLocalEnergyStack(lStack):
    energyStack = []
    for i in range(len(lStack)):
        laplacian = lStack[i]
        laplacian_squared = np.square(laplacian)
        energy = lowPass(laplacian_squared, 45, 2 ** (i+1))
        energyStack.append(energy)
    return energyStack

def warpEnergyStack(eStack, inputShape, inputTri, exampleShape):
    warpedStack = []
    for elem in eStack:
        #WARP EVERY TRIANGLE FROM EXAMPLE TRIANGULATION TO INPUT TRIANGULATION HERE
        #warped = morph(inputIm, elem, inputShape, exampleShape, 0, 1, IS_GRAY=False)
        warped = warp(elem, exampleShape, inputShape, inputTri)
        warpedStack.append(warped)
    return warpedStack

def warpLapStack(lStack, exampleShape, inputShape, inputTri):
    warpedLapStack = []
    for elem in lStack:
        warped = warp(elem, exampleShape, inputShape, inputTri)
        warpedLapStack.append(warped)
    return warpedLapStack

def robustTransfer(inputLapStack, warpedStack, inputEStack):
    newGainStack = []
    e_0 = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.9
    for i in range(len(inputLapStack)):
        gain = (warpedStack[i] / (inputEStack[i] + e_0)) ** 0.5
        gain[gain > gain_max] = gain_max
        gain[gain < gain_min] = gain_min
        showImage(gain)
        newLayer = inputLapStack[i] * gain
        newGainStack.append(newLayer)
    return newGainStack

#This is based more off of the matlab code
def styleTransfer(input, example, input_gray, example_gray):
    #Getting masks around the regions of interest
    input_mask = readGrayScale('jose_mask.jpg')
    example_mask = readGrayScale('george_mask.jpg')
    showImage(input_mask)
    showImage(example_mask)
    inputShape, inputTri = getFacialLandmarks(input)
    exampleShape, exampleTri = getFacialLandmarks(example)

    stack_depth = 6
    #The implementation given for the paper doesn't really use the Gaussian stack
    #gStackInput, gStackExample = getGaussianStacks(input_gray, example_gray, stack_depth)
    #lStackInput, lStackExample = getLaplacianStacks(gStackInput, gStackExample, input_gray, example_gray)
    lStackInput, lStackExample = getLaplacianStacks(input_gray, example_gray, input_mask, example_mask, stack_depth)

    input_residual = getResidual(input_gray, input_mask, stack_depth)
    example_residual = getResidual(example_gray, example_mask, stack_depth)

    #exampleWarpedLap = warpLapStack(lStackExample, exampleShape, inputShape, inputTri)
    #for w in exampleWarpedLap:
        #showImage(w)

    inputEStack = getLocalEnergyStack(lStackInput)
    exampleEStack = getLocalEnergyStack(lStackExample)
    #exampleEStackWarped = getLocalEnergyStack(exampleWarpedLap)

    warpedStack = warpEnergyStack(exampleEStack, inputShape, inputTri, exampleShape)

    #gainStack = robustTransfer(lStackInput, exampleEStackWarped, inputEStack)
    gainStack = robustTransfer(lStackInput, warpedStack, inputEStack)

    warpedEResidual = warp(example_residual, exampleShape, inputShape, inputTri)
    
    gainStack.append(warpedEResidual)
    output_test = sumStack(gainStack)
    showImage(output_test)
    saveImage('./test_energy_first.jpg', output_test)




input = read('jose.jpg')
example = read('george.jpg')
input_gray = readGrayScale('jose.jpg')
example_gray = readGrayScale('george.jpg')

styleTransfer(input, example, input_gray, example_gray)
