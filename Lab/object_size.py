"""
@version: v1.0
@author: Mystery Guest
@license: Apache Licence
@contact: mysticalguest@163.com
@site: 
@software: PyCharm
@file: object_size.py
@time: 2021/11/19 9:36
"""
'''
(OpenCV)图像目标尺寸检测
http://www.woshicver.com/
https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
https://blog.csdn.net/u010636181/article/details/80659700/
'''
# import the necessary packages

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# 用于计算两个（x，y）坐标之间的中点
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# 解析我们的命令行参数
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
    help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# 从磁盘加载我们的图像，将其转换为灰度，然后使用高斯过滤器平滑它
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# 然后我们执行边缘检测和扩张+磨平，以消除边缘图中边缘之间的任何间隙
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# 找到等高线，也就是我们边缘图中物体相对应的轮廓线。
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
# 然后，这些等高线区域从左到右（使得我们可以提取到参照物）在下一行代码中进行排列
# print(cnts[0])
cnts,_ = contours.sort_contours(np.array(cnts))
# 对pixelsPerMetric值进行初始化
pixelsPerMetric = None

# 下一步是对每一个等高线区域值大小进行检查校验
# loop over the contours individually
# 开始对每个单独的轮廓值进行循环
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    # 如果等高线区域大小不够大，我们就会丢弃该区域，认为它是边缘检测过程遗留下来的噪音
    if cv2.contourArea(c) < 100:
        continue

    # compute the rotated bounding box of the contour
    # 如果等高线区域足够大，我们就会在下面2行计算图像的旋转边界框，
    # 特别注意：cv2.cv.BoxPoints函数是针对于opencv2.4版本，而cv2.BoxPoints函数是针对于OpenCV 3版本
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # 然后我们将旋转的边界框坐标按顺序排列在左上角，右上角，右下角，左下角
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    # 用绿色画出物体的轮廓，然后将边界框矩形的顶点画在小的红色圆圈中。现在我们已经有了边界框，接下来就可以计算出一系列的中点
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # 将我们前面所得的有序边界框各个值拆分出来，然后计算左上角和右上角之间的中点，然后是计算左下角和右下角之间的中点
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # 在我们的图像上画出蓝色的中点，然后将各中间点用紫色线连接起来
    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)

    # 接下来，我们需要通过查看我们的参照物来初始化pixelsPerMetric变量
    # 计算中间点集之间的欧几里得距离（下2行）。
    # dA变量将包含高度距离（以像素为单位），而dB将保持我们的宽度距离
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    # 进行检查，看看我们的pixelsPerMetric变量是否已经被初始化了，如果没有，
    # 我们将dB除以我们提供的宽度，从而得到每英寸的（近似）像素。
    # 现在我们已经定义了pixelsPerMetric变量，我们可以测量图像中各物体的大小
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    # compute the size of the object
    # 计算物体的尺寸（英寸），方法是通过pixelsper度量值划分各自的欧几里得距离
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    # show the output image
    # 在我们的图像上画出物体的尺寸，而第112和113行显示输出结果
    # cv2.imshow("Image", orig)
    cv2.imshow("After Solution", orig)
    cv2.waitKey(0)