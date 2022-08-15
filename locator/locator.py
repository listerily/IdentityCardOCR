#身份证定位函数

#导包
import cv2 as cv
import numpy as np

#参数1：图片路径，参数2，图片大小归一化最大宽度
def IDCardLocate(card_file_path,MAX_WIDTH):

    #1. 装载图片
    origin_image = cv.imdecode(np.fromfile(card_file_path, dtype=np.uint8), -1)

    # cv.imshow('origin_image', origin_image)
    # cv.waitKey()
    # cv.destroyAllWindows()

    #2.图片大小归一化
    rows, cols= origin_image.shape[:2]
    if cols >  MAX_WIDTH:
            change_rate = MAX_WIDTH / cols
            sized_image = cv.resize(origin_image ,( MAX_WIDTH ,int(rows * change_rate) ), interpolation = cv.INTER_AREA)

    # cv.imshow('sized_image', sized_image)
    # cv.waitKey()
    # cv.destroyAllWindows()

    #3. 完成高斯模糊-预处理
    gaus_blured_image = cv.GaussianBlur(sized_image, (5, 5), 0)
    median_image = cv.medianBlur(gaus_blured_image,5)
    blured_image = cv.bilateralFilter(median_image, 13, 15, 15)
    blured_image = cv.bilateralFilter(blured_image, 13, 15, 15)
    # cv.imshow('blured_image',blured_image)
    # cv.waitKey()
    # cv.destroyAllWindows()

    #2. 灰度图
    gray_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)
    # gray_image = cv.cvtColor(origin_image, cv.COLOR_BGR2GRAY)

    #4.边缘检测
    canny=cv.Canny(gray_image, 40, 120)##
    # sobel = cv.Sobel(gray_image, cv.CV_8U, 1, 0, ksize=3)
    # cv.imshow('canny', canny)
    # cv.waitKey()
    # cv.destroyAllWindows()

    #5.二值化
    is_success, binary_image = cv.threshold(canny, 60, 255, cv.THRESH_OTSU)
    # cv.imshow('binary_image',binary_image)
    # cv.waitKey()
    # cv.destroyAllWindows()

    #6. 在原图上获取轮廓并绘制
    # 获取所有的轮廓，轮廓边框使用最小模式（1个像素）
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    draw_image=cv.drawContours(sized_image,contours, -1, (0, 0, 255))

    # contours
    # cv.imshow('contours_image', draw_image)
    # cv.waitKey()
    # cv.destroyAllWindows()
    #7. 基于面积（长宽比）获取卡片区域
    # 声明所有候选|满足条件的区域列表
    candidate_regions = []
    # 遍历所有的轮廓
    for i in np.arange(len(contours)):
        # 提取所有轮廓的左上坐标，及宽、高
        x, y, w, h = cv.boundingRect(contours[i])
        # 计算宽高比
        ratio = w * 1.0 / h
        # 如果是竖排情况，处理长宽比=取反
        if ratio < 1:
            ratio = 1.0 / ratio
        # 求出该区域面积
        area = w * h
        # 关键条件：宽高比在[1/1.58, 2]之间
        # 次要条件：区域面积不能太小
        if  area > 100*100 and ratio > 0.5 and ratio < 2.0:
            #按照原始大小返回图片
            candidate_regions.append(origin_image[(int)(y//change_rate):(int)((y+h)//change_rate),(int)(x//change_rate):(int)((x+w)//change_rate)])
            #按照缩放大小返回图片
            # candidate_regions.append(sized_image[y:y+h,x:x+w])
    # 如果候选区域没有数据，说明提取车牌区域失败|没有车牌区域
    if len(candidate_regions) == 0:
        print('没有找到身份证区域')
    else:
       return candidate_regions

MAX_WIDTH = 500 #设置最大宽度
card_file_path = 'images/IDCard/syx2.jpg'#传入文件路径
results = IDCardLocate(card_file_path,MAX_WIDTH)

#8. 逐个显示提取的身份证候选区域
if results:
    for i in np.arange(len(results)):
        # candidate_regions[i] 中保存的是满足条件的一个车牌区域在原始图片中的采样结果

        cv.imshow(str(i), results[i])
        cv.moveWindow(str(i),300,300)
        cv.waitKey()
        cv.destroyAllWindows()