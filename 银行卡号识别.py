import cv2
import imutils
import numpy as np
# 导入argparse模块，该模块用于解析命令行参数，方便你在运行 Python 脚本时传入不同的参数。
import argparse
# 从imutils库中导入contours模块，imutils是一个封装了许多常用图像处理功能的库，contours模块可能包含处理图像轮廓的相关函数。
from imutils import contours
import utils

# 指定图像类型
args = {
    "template": "ocr_a_reference.png",
    "image": "credit_card_01.png"
}
# 在许多图像处理或光学字符识别（OCR）相关的程序中，通常会使用字典来传递多个参数，这样可以使代码更加清晰和易于维护。
# args 字典在这里的作用就是存储程序运行所需的两个重要文件的文件名。
# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


# 绘图展示：展示窗口图像 并设置任意键退出
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 设置为0时 任意键退出
    cv2.destroyAllWindows()


##对模板进行预处理：灰度图+二值处理
# 读取一个模板文件
img = cv2.imread(args["template"])
# cv_show("img", img)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('ref', ref)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
# cv_show('ref', ref)

## 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图），此函数在opencv 4.x返回两个值
# cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留轮廓的端点
# 返回的list中每个元素都是图像中的一个轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 在图像上绘制轮廓：-1代表绘制所有轮廓
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
# cv_show('img', img)
# print(np.array(refCnts).shape)
# sort_contours为自定义排序函数
refCnts = utils.sort_contours(refCnts, method="left-to-right")[0]  # 排序从左到右，从上到下

# 创建一个字典，用于存储后续提取并处理后的图像区域（数字模板）。字典的键是索引 i，值是对应的调整大小后的图像块。
digits = {}

'''
第一个参数：img是原图
第二个参数：（x，y）是矩阵的左上点坐标
第三个参数：（x+w，y+h）是矩阵的右下点坐标
第四个参数：（0,255,0）是画线对应的rgb颜色
'''
# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    # enumerate(refCnts) 函数会为 refCnts 列表中的每个元素分配一个索引 i，并将索引和元素作为一个元组返回。
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    # 是对参考图像 ref 进行切片操作，提取出外接矩形所包含的区域，即感兴趣区域（ROI），并将其赋值给变量 roi。
    roi = cv2.resize(roi, (57, 58))
    # 使用 cv2.resize 函数将提取的 ROI 调整为固定大小 (57, 58)，即宽度为 57 像素，高度为 58 像素。
    # 调整大小的目的是为了后续处理的一致性，确保所有数字模板具有相同的尺寸

    # 每一个数字对应一个模板
    digits[i] = roi
    # 将调整大小后的 ROI 作为值，当前轮廓的索引 i 作为键，
    # 存储到字典 digits 中。这样，每个轮廓对应的数字模板就可以通过其索引进行访问

# 初始化卷积核
# cv2.getStructuringElement(shape, ksize)
# shape：指定卷积核的形状，常见取值有：
# cv2.MORPH_RECT：矩形形状。
# cv2.MORPH_ELLIPSE：椭圆形形状。
# cv2.MORPH_CROSS：十字形形状。
# ksize：指定卷积核的大小，是一个元组 (width, height)，表示卷积核的宽度和高度。

# 矩形核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# 通常这种宽而矮的矩形卷积核可以用于检测图像中水平方向的特征，比如在识别身份证号、银行卡号等数字序列时，
# 由于数字是水平排列的，使用这种矩形核可以更好地突出数字之间的水平连接关系。
# 矩形核
# 方形卷积核比较通用，在一些需要对图像进行整体形态学操作，不特别强调某个方向特征时使用，
# 例如去除图像中的小噪声点、填充小空洞等操作。
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
##对输入图像进行处理
# 读取输入图像，预处理
image = cv2.imread(args["image"])
# cv_show('image', image)

image = utils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('gray', gray)

# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# cv_show('tophat', tophat)

# 计算
# sobel算子计算水平梯度梯度 寻找边界
# gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
# np.absolute() 的局限性：
# np.absolute() 只是简单地对数组中的每个元素取绝对值，不会改变数据类型。
# 取完绝对值后，数据仍然是 float32 类型。而在图像处理中，很多操作（如显示图像）要求数据类型为 uint8（8 位无符号整数），
# 使用 np.absolute() 处理后还需要额外的步骤进行数据类型转换。
# cv2.convertScaleAbs() 的优势：
# 该函数会自动将处理后的结果转换为 uint8 类型，这是 OpenCV 中图像显示和存储常用的数据类型。
# 使用它可以一步完成取绝对值、数据类型转换和饱和处理，方便后续的图像显示和保存操作。
# gradX = np.absolute(gradX)
# (minVal, maxVal) = (np.min(gradX), np.max(gradX))
# gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
# gradX = gradX.astype("uint8")
# print(np.array(gradX).shape)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
gradX = cv2.convertScaleAbs(gradX)
# cv_show('gradX', gradX)

# 通过闭操作，（先膨胀，在腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# cv_show('gradX', gradX)
# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show('thresh', thresh)

# 再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# cv_show('thresh', thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
# cv_show('img', cur_img)


##筛选轮廓
# 创建一个空列表 locs，用于存储后续筛选出的符合条件的外接矩形的坐标信息
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    # 按长宽比例筛选
    ar = w / float(h)

    # 适合合适的区域，根据实际任务来，这里的基本是四个数字一组
    if ar > 2.5 and ar < 4.0:
        # 用长与宽筛选
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))
# 将符合的轮廓从左到右排序
# sorted(iterable, key=None, reverse=False)
# iterable：必需参数，表示要进行排序的可迭代对象，这里就是 locs 列表。
# key：可选参数，是一个函数，用于指定排序的依据。该函数会作用于可迭代对象中的每个元素，返回一个用于比较的值。
# reverse：可选参数，是一个布尔值，False 表示升序排序（默认值），True 表示降序排序。
locs = sorted(locs, key=lambda x: x[0])
# 创建一个空列表
output = []
# 创建一个空列表 output，用于存储每个外接矩形区域对应的数字信息。
# 在后续代码中，每个区域的数字信息会以某种形式（如列表）添加到 output 中
# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # 对于每个外接矩形区域，创建一个空列表 groupOutput，用于临时存储该区域内的数字信息
    groupOutput = []
    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)
    # 预处理
    # cv2.threshold 函数返回两个值，第一个值是计算得到的阈值（在使用 Otsu 算法时就是最优阈值），第二个值是二值化后的图像。
    # 这里取返回值的第二个元素 [1]，将二值化后的图像重新赋值给 group。
    # cv2.THRESH_OTSU：表示使用 Otsu 算法自动计算最优阈值。
    # Otsu 算法会遍历所有可能的阈值，找到能使类间方差最大的阈值，从而实现图像的自适应二值化
    # 使用Ostu时 阈值thresh设为0
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 计算每一个轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    # 计算每一组总的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 58))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []
        # 在模板章中计算每一个得分
        for (digit, digiROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digiROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    # cv2.rectangle 函数：这是 OpenCV 库中用于在图像上绘制矩形的函数，
    # 其基本语法为 cv2.rectangle(img, pt1, pt2, color, thickness)
    # pt1,pt2 为左上角与右下角坐标
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    # cv2.putText 函数：这是 OpenCV 库中用于在图像上添加文本的函数，
    # 其基本语法为 cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
    # org：文本的起始坐标，即文本左下角的位置。它是一个元组，格式为 (x, y)
    # fontFace：字体类型，指定了文本的字体样式。OpenCV 提供了多种预定义的字体类型。
    # cv2.FONT_HERSHEY_SIMPLEX：简单的无衬线字体，是最常用的字体之一。
    # cv2.FONT_HERSHEY_PLAIN：细的无衬线字体。
    # cv2.FONT_HERSHEY_DUPLEX：比 cv2.FONT_HERSHEY_SIMPLEX 更复杂一些的无衬线字体。
    # cv2.FONT_HERSHEY_COMPLEX：衬线字体。
    # cv2.FONT_HERSHEY_TRIPLEX：比 cv2.FONT_HERSHEY_COMPLEX 更复杂的衬线字体。
    # cv2.FONT_HERSHEY_COMPLEX_SMALL：cv2.FONT_HERSHEY_COMPLEX 的小号版本。
    # cv2.FONT_HERSHEY_SCRIPT_SIMPLEX：手写风格的字体。
    # cv2.FONT_HERSHEY_SCRIPT_COMPLEX：更复杂的手写风格字体。
    # fontScale：字体的缩放比例，用于调整文本的大小。它是一个浮点数，值越大，文本越大；值越小，文本越小
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 得到结果
    output.extend(groupOutput)
# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
# "Credit Card Type: {}".format(...)：这是 Python 的字符串格式化方法，将 FIRST_NUMBER[output[0]] 的值插入到字符串
# Credit Card Type: {} 的占位符 {} 中，然后打印出信用卡的类型信息
print("Credit Card #: {}".format("".join(output)))
# "".join(output)：join 是字符串对象的方法，用于将列表中的元素连接成一个字符串。
# 这里将 output 列表中的所有数字字符连接成一个完整的信用卡号码字符串
# "Credit Card #: {}".format(...)：同样使用字符串格式化方法，将连接好的信用卡号码插入到字符串
# Credit Card #: {} 的占位符中，然后打印出完整的信用卡号码
cv2.imshow("Image", image)
cv2.waitKey(0)
