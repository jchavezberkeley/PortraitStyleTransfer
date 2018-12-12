import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import cv2
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
        energy = lowPass(laplacian_squared, 5*(2**(i+1)), 2**(i+1))
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

#Alternate approach of warping the Laplacian stacks before estimating energy
def warpLapStack(lStack, exampleShape, inputShape, inputTri):
    warpedLapStack = []
    for elem in lStack:
        warped = warp(elem, exampleShape, inputShape, inputTri)
        warpedLapStack.append(warped)
    return warpedLapStack

#Performs Robust transfer and gain clamping
def robustTransfer(inputLapStack, warpedStack, inputEStack):
    newGainStack = []
    e_0 = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.9
    for i in range(len(inputLapStack)):
        gain = (warpedStack[i] / (inputEStack[i] + e_0)) ** 0.5
        gain[gain > gain_max] = gain_max
        gain[gain < gain_min] = gain_min
        gain = lowPass(gain, 5*(2**i), 3*(2**i))
        newLayer = inputLapStack[i] * gain
        newGainStack.append(newLayer)
    return newGainStack

def configureBackground(image, mask, im2name):
    mask = np.bitwise_or(np.roll(mask, 6, axis=1), mask)
    mask = np.bitwise_or(np.roll(mask, -6, axis=1), mask)
    mask = np.bitwise_or(np.roll(mask, 6, axis=0), mask)
    mask = np.bitwise_or(np.roll(mask, -6, axis=0), mask)

    background = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    saveImage('./images/' + im2name + '_background.jpg', background)

#This is based more off of the matlab code
def styleTransfer(input, example, input_mask, example_mask, inputShape, exampleShape, input_channel, example_channel):
    inputShape, inputTri = getFacialLandmarks(input)
    exampleShape, exampleTri = getFacialLandmarks(example)
    inputTri = scipy.spatial.Delaunay(inputShape)
    exampleTri = scipy.spatial.Delaunay(exampleShape)

    stack_depth = 6

    lStackInput, lStackExample = getLaplacianStacks(input_channel, example_channel, input_mask, example_mask, stack_depth)

    input_residual = getResidual(input_channel, input_mask, stack_depth)
    example_residual = getResidual(example_channel, example_mask, stack_depth)

    inputEStack = getLocalEnergyStack(lStackInput)
    exampleEStack = getLocalEnergyStack(lStackExample)

    warpedStack = warpEnergyStack(exampleEStack, inputShape, inputTri, exampleShape)

    gainStack = robustTransfer(lStackInput, warpedStack, inputEStack)
    warpedEResidual = warp(example_residual, exampleShape, inputShape, inputTri)

    gainStack.append(warpedEResidual)
    output = sumStack(gainStack)
    return rescale(output)

imname = sys.argv[1]
im2name = sys.argv[2]
gray = True if sys.argv[3].lower() == 'true' else False
outname = sys.argv[4]

folder = './images/'
file_type = '.jpg'
_mask = '_mask'
_background = '_background'

input = read(folder + imname + file_type)
example = read(folder + im2name + file_type)
input_colors = readColors(folder + imname + file_type)
example_colors = readColors(folder + im2name + file_type)
input_gray = readGrayScale(folder + imname + file_type)
example_gray = readGrayScale(folder + im2name + file_type)
input_mask_gray = readGrayScale(folder + imname + _mask + file_type)
example_mask_gray = readGrayScale(folder + im2name + _mask + file_type)
input_mask = cv2.imread(folder + imname + _mask + file_type, 0)
example_mask = cv2.imread(folder + im2name + _mask + file_type, 0)
inputShape = np.loadtxt('./points/A_points_jose.txt')
exampleShape = np.loadtxt('./points/A_points_chris.txt')


configureBackground(example, example_mask, im2name)

if gray:
    background_colors = readGrayScale(folder + im2name + _background + file_type)
    gray = styleTransfer(input, example, input_mask_gray, example_mask_gray, inputShape, exampleShape, input_gray, example_gray)
    gray = (background_colors * (1-input_mask_gray)) + (gray * input_mask_gray)
    output = gray
else:
    background_colors = readColors(folder + im2name + _background + file_type)
    background_red = background_colors[0]
    background_green = background_colors[1]
    background_blue = background_colors[2]

    red = styleTransfer(input, example, input_mask_gray, example_mask_gray, inputShape, exampleShape, input_colors[0], example_colors[0])

    green = styleTransfer(input, example, input_mask_gray, example_mask_gray, inputShape, exampleShape, input_colors[1], example_colors[1])

    blue = styleTransfer(input, example, input_mask_gray, example_mask_gray, inputShape, exampleShape, input_colors[2], example_colors[2])

    red = (background_red * (1-input_mask_gray)) + (red * input_mask_gray)
    green = (background_green * (1-input_mask_gray)) + (green * input_mask_gray)
    blue = (background_blue * (1-input_mask_gray)) + (blue * input_mask_gray)
    output = np.dstack([red, green, blue])
showImage(output)
saveImage('./' + outname + file_type, output)
