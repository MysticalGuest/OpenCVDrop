"""
@version: v1.0
@author: Mystery Guest
@license: Apache Licence
@contact: mysticalguest@163.com
@site: 
@software: PyCharm
@file: try.py
@time: 2021/11/30 15:00
"""

import cv2 as cv
import numpy as np

'''
imread(img_path,flag) 读取图片，返回图片对象
    img_path: 图片的路径，即使路径错误也不会报错，但打印返回的图片对象为None
    flag：cv2.IMREAD_COLOR，读取彩色图片，图片透明性会被忽略，为默认参数，也可以传入1
          cv2.IMREAD_GRAYSCALE,按灰度模式读取图像，也可以传入0
          cv2.IMREAD_UNCHANGED,读取图像，包括其alpha通道，也可以传入-1
'''

img = cv.imread(r"target.tif")
# img1 = cv.imread(r"target.tif", 0)
# print(img.shape)
# cv.imshow("img1", img1)
# 该函数将输入图像从一个色彩空间转换为另一个色彩空间
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
'''
ret, dst = cv2.threshold(src, thresh, maxval, type)
dst： 输出图
thresh： 阈值
maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； 
    cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
'''
ret, img_threshold = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
cv.imshow("img_gray", img_gray)
'''
dst = cv2.adaptiveThreshold(src, maxval, thresh_type, type, Block Size, C)
maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
thresh_type： 阈值的计算方法，包含以下2种类型：
    cv2.ADAPTIVE_THRESH_MEAN_C均值
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C加权
type：二值化操作的类型，与固定阈值函数相同，包含以下5种类型： cv2.THRESH_BINARY
    cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO
    cv2.THRESH_TOZERO_INV.
Block Size： 图片中分块的大小
C ：阈值计算方法中的常数项
'''
# cv2.ADAPTIVE_THRESH_MEAN_C : 阈值是邻域面积的平均值
th1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                           cv.THRESH_BINARY, 11, 2)
cv.imshow("th1", th1)

# 高斯模糊
'''
GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None):
@param ksize 高斯核大小。 ksize.width 和 ksize.height 可以不同，但它们都必须是正数和奇数。 或者，它们可以是零，然后根据 sigma 计算它们。
@param sigmaX X 方向的高斯核标准偏差。
'''
gb = cv.GaussianBlur(th1, (9, 9), 0)
# cv.imshow("gb", gb)
'''
Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
@param threshold1 滞后过程的第一个阈值。
@param threshold2 滞后过程的第二个阈值。
'''
canny = cv.Canny(gb, 50, 150)
# cv.imshow("canny", canny)
# 轮廓检测函数
'''
findContours(image, mode, method, contours=None, hierarchy=None, offset=None):
@param image Source，8 位单通道图像。非零像素被视为 1。零像素保持 0，
    因此图像被视为 binary 。
    您可以使用 #compare、#inRange、#threshold、#adaptiveThreshold、
    #Canny 和其他工具从灰度或彩色图像中创建二进制图像。
    如果 mode 等于 #RETR_CCOMP 或 #RETR_FLOODFILL，
    则输入也可以是标签的 32 位整数图像 (CV_32SC1)。
@param contours 检测到的轮廓。
    每个轮廓都存储为一个点向量（例如 std::vector<std::vector<cv::Point> >）。
@param hierarchy 可选输出向量（例如 std::vector<cv::Vec4i>），
    包含有关图像拓扑的信息。它的元素与轮廓的数量一样多。对于每个第 i 个轮廓轮廓[i]，
    元素hierarchy[i][0]、hierarchy[i][1]、hierarchy[i][2]和
    hierarchy[i][3]被设置为0-基于相同层次级别的下一个和前一个轮廓的轮廓中的索引，
    分别是第一个子轮廓和父轮廓。如果轮廓 i 没有下一个、上一个、父级或嵌套轮廓，
    则层级 [i] 的相应元素将为负。
@param mode 轮廓检索模式，见#RetrievalModes
    1.RETR_EXTERNAL；只提取整体外部轮廓；
    2.RETR_LIST； 提取所有轮廓，不需要建立任何继承关系；
    3.RETR_CCOMP ；提取所有轮廓，最后形成连个水平集，外面一个，内部一个；
    4.RETR_TREE ；提取所有轮廓，构建等级关系（父子继承关系）
@param method 轮廓逼近方法，见#ContourApproximationModes
'''
image, contours, hierarchy = cv.findContours(gb, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(contours)
img_copy = np.zeros(canny.shape, np.uint8)
'''
drawContours(image, contours, contourIdx, color,thickness)
    image 绘制轮廓的图像 ndarray 格式；
    contours ,findContours 函数找到的轮廓列表；
    contourIdx 绘制轮廓的索引数，取整数时绘制特定索引的轮廓，为负值时，绘制全部轮廓；
    color 绘制轮廓所用到的颜色，这里需要提醒一下， 想使用 RGB 彩色绘制时，
        必须保证 输入的 image 为三通道，否则轮廓线非黑即白；
    thickness ，用于绘制轮廓线条的宽度，取负值时将绘制整个轮廓区域；
'''
cv.drawContours(img_copy, contours, -1, (255, 255, 255), 1)
# cv.drawContours(img_copy, contours, -3, (0, 255, 0), 1)
# cv.imshow("After contour recognition", img_copy)
# print(contours[1])
# [[[1037 1198]]
#
#  [[1037 1199]]
#
#  [[1038 1198]]
#
#  [[1041 1198]]
#
#  [[1042 1199]]
#
#  [[1043 1199]]
#
#  [[1042 1198]]]
# temp = [[[0,0]],[[0,0]],[[0,0]][[0,0]],[[0,0]],[[0,0]],[[0,0]]]
i=0
co=[]
# print(np.ones(contours[0].shape, dtype = np.uint8) * 255)
for contour in contours:
    area = cv.contourArea(contour)
    arc = cv.arcLength(contour, False)
    # print(arc)
    # print(area)
    if arc>50 and area>5:
        # print(area)
        # print(i)
        # contours[i]=temp
        # print(type(np.zeros(contour.shape, np.uint8)))
        # contours[i]=np.zeros(contour.shape, np.uint8)
        # co.append(np.zeros(contour.shape, np.uint8))
        # co.append(np.ones(contour.shape, dtype = np.uint8) * 255)
    # else:
        co.append(contour)
    i+=1
img_copy_after_judge = np.zeros(canny.shape, np.uint8)
# print(contours[10].shape)
# print(co[10].shape)
# print(contours[10])
# print(co[10])
# print(type(contours[10]))
# print(type(co[10]))
# print(type(contours))
# print(type(co))
cv.drawContours(img_copy_after_judge, co, -1, (255, 255, 255), 1)
cv.imshow("After contour screening", img_copy_after_judge)
for c in co:
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(img_copy_after_judge, (x, y), (x+w, y+h), (255, 255, 0), thickness=2)
cv.imshow("Rectangle", img_copy_after_judge)
key = cv.waitKey(0)
cv.destroyAllWindows()