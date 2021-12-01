"""
@version: v1.0
@author: Mystery Guest
@license: Apache Licence
@contact: mysticalguest@163.com
@site: 
@software: PyCharm
@file: experiment.py
@time: 2021/11/30 15:31
"""


import cv2 as cv
import numpy as np


img = cv.imread(r"target.tif")
# 灰度化
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 自适应阈值分割
th = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                           cv.THRESH_BINARY, 11, 2)
# cv.imshow("th", th)
# 高斯模糊
gb = cv.GaussianBlur(th, (9, 9), 0)

canny = cv.Canny(img, 50, 150)

# 轮廓检测函数
image, contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print(type(contours[0]))

print(type(np.zeros(canny.shape, np.uint8)))
img_copy = np.zeros(canny.shape, np.uint8)
cv.drawContours(img_copy, contours, -1, (255, 255, 255), 1)

i = 0
co = []

for contour in contours:
    area = cv.contourArea(contour)
    arc = cv.arcLength(contour, False)

    if arc>50 and area>5:

        co.append(contour)
    i += 1
img_copy_after_judge = np.zeros(canny.shape, np.uint8)

cv.drawContours(img_copy_after_judge, co, -1, (255, 255, 255), 1)
cv.imshow("After contour screening", img_copy_after_judge)
for c in co:
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(img_copy_after_judge, (x, y), (x+w, y+h), (255, 255, 0), thickness=1)
cv.imshow("Rectangle", img_copy_after_judge)
key = cv.waitKey(0)
cv.destroyAllWindows()
