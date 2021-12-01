"""
@version: v1.0
@author: Mystery Guest
@license: Apache Licence
@contact: mysticalguest@163.com
@site: 
@software: PyCharm
@file: learn2.py
@time: 2021/11/17 21:01
"""

# 图像像素获取和编辑
import cv2
img = cv2.imread(r"dog.jpg")
# 返回(1200, 1920, 3), 分辨率, 宽1200像素(rows)，长1920像素(cols)，3通道(channels)
print(img.shape)
# 返回6912000，所有像素数量，=1200*1920*3
print(img.size)
# type
print(img.dtype)   # dtype('uint8')

# 获取和设置
pixel = img[100, 100]  # [57 63 68],获取(100,100)处的像素值
print(pixel)
img[100, 100] = [57, 63, 99]  # 设置像素值
b = img[100, 100, 0]    # 57, 获取(100,100)处，blue通道像素值
g = img[100, 100, 1]    # 63
r = img[100, 100, 2]      # 68
r = img[100, 100, 2]=99    # 设置red通道值

# 获取和设置
piexl = img.item(100, 100, 2)
img.itemset((100, 100, 2), 99)

