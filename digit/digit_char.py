#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys

from PIL import Image, ImageFont, ImageDraw
import argparse
from argparse import RawTextHelpFormatter
import csv
import os
import cv2
import random
import numpy as np
import traceback
import copy


class dataAugmentation(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    @classmethod
    def add_noise(cls, img):
        for i in range(20):  # 噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x, temp_y] = 255
        return img

    @classmethod
    def add_erode(cls, img):  # 腐蚀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(img, kernel)
        return img

    @classmethod
    def add_dilate(cls, img):  # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel)
        return img

    def do(self, img_list=[]):
        aug_list = copy.deepcopy(img_list)
        for i in range(len(img_list)):
            im = img_list[i]
            if self.noise and random.random() < 0.5:
                im = self.add_noise(im)
            if self.dilate and random.random() < 0.5:
                im = self.add_dilate(im)
            elif self.erode:
                im = self.add_erode(im)
            aug_list.append(im)
        return aug_list


class FindImageBBox(object):
    def __init__(self, ):
        pass

    def do(self, img):
        height = img.shape[0]
        width = img.shape[1]
        v_sum = np.sum(img, axis=0)
        h_sum = np.sum(img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1
        # 从左往右扫描，遇到非零像素点就以此为字体的左边界
        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break
        # 从右往左扫描，遇到非零像素点就以此为字体的右边界
        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break
        # 从上往下扫描，遇到非零像素点就以此为字体的上边界
        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        # 从下往上扫描，遇到非零像素点就以此为字体的下边界
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        return (left, top, right, low)


# 把字体图像放到背景图像中		
class PreprocessResizeKeepRatioFillBG(object):

    def __init__(self, width, height,
                 fill_bg=False,
                 auto_avoid_fill_bg=True,
                 margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin

    @classmethod
    def is_need_fill_bg(cls, cv2_img, th=0.5, max_val=255):
        image_shape = cv2_img.shape
        height, width = image_shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False

    @classmethod
    def put_img_into_center(cls, img_large, img_small):
        width_large = img_large.shape[1]
        height_large = img_large.shape[0]

        width_small = img_small.shape[1]
        height_small = img_small.shape[0]

        if width_large < width_small:
            raise ValueError("width_large <= width_small")
        if height_large < height_small:
            raise ValueError("height_large <= height_small")

        start_width = (width_large - width_small) // 2
        start_height = (height_large - height_small) // 2

        img_large[start_height:start_height + height_small,
        start_width:start_width + width_small] = img_small
        return img_large

    def do(self, cv2_img):
        # 确定有效字体区域，原图减去边缘长度就是字体的区域
        if self.margin is not None:
            width_minus_margin = max(2, self.width - self.margin)
            height_minus_margin = max(2, self.height - self.margin)
        else:
            width_minus_margin = self.width
            height_minus_margin = self.height

        if len(cv2_img.shape) > 2:
            pix_dim = cv2_img.shape[2]
        else:
            pix_dim = None

        if self.auto_avoid_fill_bg:
            need_fill_bg = self.is_need_fill_bg(cv2_img)
            if not need_fill_bg:
                self.fill_bg = False
            else:
                self.fill_bg = True

        # should skip horizontal stroke
        if not self.fill_bg:
            ret_img = cv2.resize(cv2_img, (width_minus_margin,
                                           height_minus_margin))
        else:
            if pix_dim is not None:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin),
                                    np.uint8)
            # 将缩放后的字体图像置于背景图像中央
            ret_img = self.put_img_into_center(norm_img, cv2_img)

        if self.margin is not None:
            if pix_dim is not None:
                norm_img = np.zeros((self.height,
                                     self.width,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((self.height,
                                     self.width),
                                    np.uint8)
            ret_img = self.put_img_into_center(norm_img, ret_img)
        return ret_img


# 检查字体文件是否可用
class FontCheck(object):

    def __init__(self, lang_chars, width=46, height=46):
        self.lang_chars = lang_chars
        self.width = width
        self.height = height

    def do(self, font_path):
        width = self.width
        height = self.height
        try:
            for i in range(len(self.lang_chars)):
                # 白色背景
                img = Image.new("RGB", (width, height), (0, 0, 0))
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype(font_path, int(width * 0.9))
                # 黑色字体
                draw.text((0, 0), self.lang_chars[i], 255,
                          font=font)
                data = list(img.getdata())
                sum_val = 0
                for i_data in data:
                    sum_val += sum(i_data)
                if sum_val < 2:
                    return False
        except:
            print("fail to load:%s" % font_path)
            traceback.print_exc(file=sys.stdout)
            return False
        return True


# 生成字体图像
class Font2Image(object):

    def __init__(self, width, height, need_crop, margin):
        self.width = width
        self.height = height
        self.need_crop = need_crop
        self.margin = margin

    def do(self, font_path, char, rotate=0):
        find_image_bbox = FindImageBBox()
        # 白色背景
        img = Image.new("RGB", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width*1.2))
        # 黑色字体
        draw.text((5, 0), char, 255, font=font)
        if np.random.randint(0, 2):
            for j in range(3):  # 条纹
                temp_x = np.random.randint(10, img.width - 10)
                temp_y = np.random.randint(10, img.height - 10)
                temp_a = np.random.choice((-1, 1))
                shape = [(temp_x, temp_y), (temp_x + 4 * temp_a, temp_y + 4 * temp_a)]
                draw.line(shape, fill="white", width=0)
        if rotate != 0:
            img = img.rotate(rotate)
        data = list(img.getdata())
        sum_val = 0
        for i_data in data:
            sum_val += sum(i_data)
        if sum_val > 2:
            np_img = np.asarray(data, dtype='uint8')
            np_img = np_img[:, 0]
            np_img = np_img.reshape((self.height, self.width))
            cropped_box = find_image_bbox.do(np_img)
            left, upper, right, lower = cropped_box
            np_img = np_img[upper: lower + 1, left: right + 1]
            if not self.need_crop:
                preprocess_resize_keep_ratio_fill_bg = \
                    PreprocessResizeKeepRatioFillBG(self.width, self.height,
                                                    fill_bg=False,
                                                    margin=self.margin)
                np_img = preprocess_resize_keep_ratio_fill_bg.do(
                    np_img)
            return np_img
        else:
            print("img doesn't exist.")


def get_label_dict():
    filename = './digit.csv'
    label_dict = []
    with open(filename, 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            label_dict.append(row[0])  # 选择某一列加入到data数组中
    return label_dict


def create_img(char_list, font2image, verified_font_paths, rotate, all_rotate_angles, test_ratio):
    test_digit = {}
    train_digit = {}
    for i in range(11):  # 外层循环是字
        image_list = []
        for j, verified_font_path in enumerate(verified_font_paths):  # 内层循环是字体
            if rotate == 0:
                image = font2image.do(verified_font_path, char_list[i])
                image_list.append(image)
            else:
                for k in all_rotate_angles:
                    image = font2image.do(verified_font_path, char_list[i], rotate=k)
                    image_list.append(image)

        data_aug = dataAugmentation()
        image_list = data_aug.do(image_list)

        test_num = int(len(image_list) * test_ratio)
        random.shuffle(image_list)  # 图像列表打乱
        for j in range(len(image_list)):
            image_list[j] = 255 - image_list[j]

        t1 = image_list[:test_num]
        t2 = image_list[test_num:]
        test_digit[char_list[i]] = t1
        train_digit[char_list[i]] = t2
    return train_digit, test_digit


def create_digit():

    font_dir = os.path.expanduser('./digit_fonts')
    test_ratio = 0.2
    width = 44
    height = 44
    need_crop = False
    margin = 0
    rotate = 15
    rotate_step = 1

    # 将汉字的label读入，得到（ID：汉字）的映射表label_dict
    char_list = get_label_dict()

    char_list = char_list[2:]
    for i in range(len(char_list)):
        print(i, char_list[i])

    font_check = FontCheck(char_list)

    rotate = abs(rotate)
    all_rotate_angles = []
    if 0 < rotate <= 30:
        for i in range(0, rotate + 1, rotate_step):
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)

    # 对于每类字体进行小批量测试
    verified_font_paths = []
    # search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)
            verified_font_paths.append(path_font_file)

    font2image = Font2Image(width, height, need_crop, margin)

    train_digit, test_digit = create_img(char_list, font2image, verified_font_paths, rotate, all_rotate_angles, test_ratio)
    return train_digit, test_digit
