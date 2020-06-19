import cv2
import numpy as np

import random
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('bologna26_02_84.mkv')
#Global Variables
y = 0
x = 0
h = 200
w = 200

# rescale frame
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_720p()
change_res(1280, 720)

def salt(image, number):
    rows, cols = image.shape
    saltImage = np.copy(image)

    for i in range(number):
        randR = random.randint(0, rows - 1)
        randC = random.randint(0, cols - 1)
        saltImage[randR][randC] = 255
    return saltImage

while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (580, 400), fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # conversion of BGR to grayscale is necessary to apply this operation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # adaptive thresholding to use different threshold
    # values on different regions of the frame.
    Thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(Thresh, kernel, iterations = 1)

    dilation = cv2.dilate(Thresh, kernel, iterations=1)

    # Display Threshold
    cv2.imshow('Thresh', Thresh)

    simg = salt(Thresh, 100)




    #Gaussian filter to remove noise

    dst = cv2.GaussianBlur(simg,(5,5),cv2.BORDER_DEFAULT)

    # perform coordinates cropping image
    crop_img = dst[y:y + h, x: x + w]


    # Performance of Canny Edge on cropped frame
    edges = cv2.Canny(crop_img, 100, 200)
    
    cv2.imshow("Eroson", erosion)

    cv2.imshow("Dilation", dilation)

    cv2.imshow('Noise_Image', simg)

    cv2.imshow("mew_image", dst)
    cv2.imshow("Crop_Image", crop_img)
    cv2.imshow("Canny Edge", edges)
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()




