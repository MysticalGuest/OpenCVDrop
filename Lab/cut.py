"""
@version: v1.0
@author: Mystery Guest
@license: Apache Licence
@contact: mysticalguest@163.com
@site: 
@software: PyCharm
@file: cut.py
@time: 2021/11/19 19:46
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('target.tif')
rows, cols, ch = img.shape
pts1 = np.float32([[56, 65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0, 0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()