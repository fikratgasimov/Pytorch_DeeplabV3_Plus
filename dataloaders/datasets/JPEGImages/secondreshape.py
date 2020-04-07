import array as rr
import random
import cv2
from PIL import Image
import PIL
import os
from math import sqrt
import numpy
import math
import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
from array import *
import re

# Cutting images by white pixels

count = 1

for root, dirs, files in os.walk('/home/fikrat//Ground_Truth-JPEGImages/JPEGImages'):

    for files in sorted(os.listdir(os.getcwd())):

        if files.endswith(".jpg") and files == "Org_Img" + str(count) + ".jpg":
            print(files)

            image = cv2.imread(files, cv2.IMREAD_COLOR)


            center_height = 512
            cut_height = image.shape[0]
            # arrays = []
            # pixel = [[], []]
            #
            # whitepixel = [0, 0, 0]
            #
            # for height in range(image.shape[0]):
            #     for width in range(image.shape[1]):
            #         pixel = image[height, width]
            #         if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            #
            #             if height < cut_height:
            #                 cut_height = height

            cropped_image = image[0:center_height, 800:2200]
            path = 'Again'

            cv2.imwrite(os.path.join(path, files), cropped_image)
            count = count + 1









