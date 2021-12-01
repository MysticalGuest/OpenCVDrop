"""
@version: v1.0
@author: Mystery Guest
@license: Apache Licence
@contact: mysticalguest@163.com
@site: 
@software: PyCharm
@file: learn1.py
@time: 2021/11/17 20:48
"""

# coding:utf-8
# 图像读取和写入
import cv2
print(cv2.__version__)
img = cv2.imread(r"dog.jpg")
print(img.shape)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("img", img)
cv2.imshow("thre", img_threshold)

key = cv2.waitKey(0)
if key == 27:  # 按esc键时，关闭所有窗口
    print(key)
    cv2.destroyAllWindows()
cv2.imwrite(r"dog_out.jpg",img_threshold)