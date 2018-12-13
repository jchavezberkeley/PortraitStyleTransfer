# Portrait Style Transfer #
CS194-26 Computational Photography Final Project, based off of the SIGGRAPH paper "Style Transfer for Headshot Portraits"

Website: [https://jchavezberkeley.github.io/PortraitStyleTransfer/](https://jchavezberkeley.github.io/PortraitStyleTransfer/)

Written by Jose Chavez and Daniel Li

Link to [paper](https://people.csail.mit.edu/yichangshih/portrait_web/)
Link to [OpenCV/dlib program for automatically finding correspondence points](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

To be able to run this code, please make an `images/` and a `points/` folder in your directory with the following python files. Arguments passed in will look for images and data in these folders.

### inputs.py ###
This file opens up each image, using `ginput` and allows to click and select locations for correspondence points.

Arguments:
1. Name of first image to choose points on. Do not include file extension.
2. Name of second image to choose points on. Do not include file extension.
3. Number of points to be selected on each image.

Example: `python3 ./inputs.py jose george 43`

Outputs:
* Saves the data from the points into `.txt`, placed in the `points/` folder.

### make_mask.py ###
This file opens up an image and allows you to draw and save a binary mask. The code written in this file is not our own. The instructions to use this code is found [here](https://github.com/nikhilushinde/cs194-26_proj3_2.2)

Arguments:
1. Entire path of image to draw mask around

Example: `python3 ./make_mask.py ./jose.jpg`

### functions.py ###
This file contains all commonly used functions. They mostly deal with using skimage, specific numpy operations, functions to make Laplacian and Gaussian stacks, and our warping function.

### main.py ###
This file contains out algorithm for style transfer.

Arguments:
1. Name of input image, the image to have style transfer to it. Do not include the file extension, please put this image in `images/` folder.
2. Name of example image, the image to transfer style from. Do not include the file extension, please put this image in `images/` folder.
3. Boolean value, either True or False, on whether you want result to be in grayscale.
4. Boolean value, either True or False, on whether you want to use binary mask in the Laplacian stacks
5. Name of file to be outputted. Do not include extension.

Example: `python3 ./main.py jose george false true jose_george_test`
