import cv2
import sys
import numpy as np
import math
import numpy as np

import skimage as sk
import skimage.io as skio

# global variables for drawing on mask
drawing = False
polygon = False
centerMode = False
contours = []
polygon_center = None
img = None

def create_mask(imname):
    masks_to_ret = {"centers":[], "contours":[], "offsets":[]}

    global drawing, polygon, contours, centerMode, polygon_center
    pressed_key = 0
    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global drawing, centerMode, polygon, pressed_key
        if drawing == True and event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.circle(mask,(x,y),10,(255,255,255),-1)
        if polygon == True and event == cv2.EVENT_LBUTTONDOWN:
            contours.append([x,y])
            cv2.circle(img,(x,y),2,(255,255,255),-1)
        if centerMode == True and event == cv2.EVENT_LBUTTONDOWN:
            polygon_center = (x,y)
            print(polygon_center)
            cv2.circle(img, polygon_center, 3, (255, 0, 0), -1)
            centerMode = False

            masks_to_ret["centers"].append(polygon_center)
            masks_to_ret["contours"].append(contours)

    # Create a black image, a window and bind the function to window
    orig_img = cv2.imread(imname)
    reset_orig_img = orig_img[:]
    mask = np.zeros(orig_img.shape, np.uint8)
    img = np.array(orig_img[:])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)

    cv2.setMouseCallback('image',draw_circle)

    angle = 0
    delta_angle = 5
    resize_factor = 1.1
    total_resize = 1
    adjusted = False

    while(1):
        cv2.imshow('image',img)
        pressed_key = cv2.waitKey(20) & 0xFF

        """
        Commands:
        d: toggle drawing mode
        p: toggle polygon mode
        q: draw polygon once selected, and select center
        """

        if pressed_key == 27:
            break
        elif pressed_key == ord('d'):
            drawing = not drawing
            print("drawing status: ", drawing)
        elif pressed_key == ord('p'):
            polygon = not polygon
            print("polygon status: ", polygon)
        elif polygon == True and pressed_key == ord('q') and len(contours) > 2:
            contours = np.array(contours)
            cv2.fillPoly(img, pts=[contours], color = (255,255,255))
            cv2.fillPoly(mask, pts=[contours], color = (255,255,255))

            centerMode = True
            polygon = False
        elif pressed_key == ord('o'):
            # loop over the rotation angles again, this time ensuring
            # no part of the image is cut off
            angle = (angle + delta_angle) % 360
            adjusted = True
            print("Rotate")

        elif pressed_key == ord('i'):
            # loop over the rotation angles again, this time ensuring
            # no part of the image is cut off
            angle = (angle - delta_angle) % 360
            adjusted = True
            print("Rotate")

        # Plus
        elif pressed_key == ord('='):
            total_resize = total_resize*resize_factor
            adjusted = True
            print("Resize up")

        # Minus
        elif pressed_key == ord('-'):
            total_resize = total_resize*(1/resize_factor)
            adjusted = True
            print("Resize down")


        elif pressed_key == ord('r'):
            img = np.array(reset_orig_img)
            contours = []
            masks_to_ret["centers"] = []
            masks_to_ret["contours"] = []

            centerMode = False
            polygon = False
            angle = 0
            total_resize = 1

            print("polygon status: False")

        # adjust
        if adjusted:
            rows,cols,_ = orig_img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img = cv2.resize(orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
            img = cv2.warpAffine(img,M,(cols,rows))
            cv2.imshow('image', img)
            adjusted = False


    cv2.destroyAllWindows()
    name = imname.split('/')[-1]

    # store offsets to allow recreation of masks in target image
    for center_num in range(len(masks_to_ret["centers"])):
        offset = []
        center = masks_to_ret["centers"][center_num]
        for point in masks_to_ret["contours"][center_num]:
            xoffset = point[0] - center[0]
            yoffset = point[1] - center[1]

            offset.append([xoffset, yoffset])
        masks_to_ret["offsets"].append(offset)

    # adjust the output image
    rows,cols,_ = orig_img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    adj_orig_img = cv2.resize(reset_orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
    adj_orig_img = cv2.warpAffine(adj_orig_img,M,(cols,rows))

    return masks_to_ret, adj_orig_img

# run with 2 image names to generate and save masks and new source image
def save_mask(path):
    mask_info, input_im = create_mask(path)
    # im1 is the input, im2 is the example
    input_mask = np.zeros((input_im.shape[0], input_im.shape[1], 3))
    cv2.fillPoly(input_mask, np.array([mask_info["contours"][0]]), (255,255,255))

    name1 = path.split('/')[-1]
    name1 = name1[:-4]

    input_mask = np.clip(sk.img_as_float(input_mask), -1, 1)

    skio.imsave("./" + name1 + "_mask.jpg", input_mask)
    return input_mask

path = sys.argv[1]
save_mask(path)
