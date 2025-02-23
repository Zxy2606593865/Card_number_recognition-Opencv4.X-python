import cv2


# 参数一:list结构的，轮廓信息
# 参数二:要使用的方法
# 返回值:处理过后的轮廓和矩形轮廓
def sort_contours(cnts, method="left-to-right"):
    # method：这是一个可选参数，用于指定排序的方法，默认值为 "left-to-right"。它可以取以下几种值：
    # "left-to-right"：从左到右排序。
    # "right-to-left"：从右到左排序。
    # "top-to-bottom"：从上到下排序。
    # "bottom-to-top"：从下到上排序
    reverse = False
    # reverse：一个布尔变量，用于控制排序的顺序。
    # 如果为 True，则进行降序排序；
    # 如果为 False，则进行升序排序。初始值为 False
    i = 0
    # i=一个整数变量，用于指定排序时所依据的外接矩形的坐标索引。i = 0 表示按 x 坐标排序，i = 1 表示按 y 坐标排序。

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    '''
    cv2.boundingRect(c) 
    返回四个值，分别是x，y，w，h；
    x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    '''
    # cv2.boundingRect(cnt)绘制轮廓的外接矩形 该函数返回四个值 x，y，w，h：(x,y)为外接矩形左上角坐标
    # w,h 为外接矩形的宽和高
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 在轮廓信息中找到一个外接矩形

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # sorted() 函数用于对可迭代对象进行排序，返回一个新的已排序列表。
    # key=lambda b: b[1][i]
    # 是一个匿名函数，作为排序的依据。b 代表 zip(cnts, boundingBoxes) 生成的每个元组，b[1] 表示元组中的第二个元素，
    # 即外接矩形元组 (x, y, w, h)，b[1][i] 则表示外接矩形元组中的第 i 个元素。
    # i 的值在之前的代码中根据 method 参数确定，
    # i = 0 表示按 x 坐标排序，i = 1 表示按 y 坐标排序。
    return cnts, boundingBoxes


# 重置大小，用于比较模板和图像中的数字是否一致
# 插值方法如下：
# INTER_NEAREST:最邻近插值
# INTER_LINEAR:双线性插值,默认情况下使用该方式进行插值.
# INTER_AREA:基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,它可以产生更少的波纹,
# 但是当图像放大时,它的效果与INTER_NEAREST效果相似.
# INTER_CUBIC:4×4邻域双3次插值
# INTER_LANCZOS4:8×8邻域兰索斯插值
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]  # (200,300,3)
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # cv2.resize() 是 OpenCV 库中用于调整图像尺寸的函数。
    # interpolation = inter：这个参数指定了在调整图像尺寸过程中所使用的插值方法。inter是从resize函数的参数传递过来的，
    # 默认值为：cv2.INTER_AREA。
    # 不同的插值方法适用于不同的场景：
    # cv2.INTER_AREA：适用于缩小图像的情况，能在缩小图像时尽量保持图像的质量，减少失真。
    # cv2.INTER_LINEAR：这是默认的插值方法，适用于放大图像，计算速度较快，能提供较好的效果。
    # cv2.INTER_CUBIC：也是用于放大图像的插值方法，计算精度比
    # cv2.INTER_LINEAR
    # 更高，但速度相对较慢，能得到更清晰的放大效果。
    # cv2.INTER_NEAREST：最近邻插值方法，计算最简单，速度最快，但可能会导致图像出现锯齿状边缘，质量相对较差。
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
