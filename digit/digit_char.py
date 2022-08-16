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

DATA_NUM = 200

class dataAugmentation(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    @classmethod
    def add_noise(cls, img):
        for i in range(10):  # 噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x, temp_y] = 255
        return img

    @classmethod
    def add_erode(cls, img):  # 腐蚀
        i = np.random.randint(1, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))
        img = cv2.erode(img, kernel)
        return img

    @classmethod
    def add_dilate(cls, img):  # 膨胀
        i = np.random.randint(1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))
        img = cv2.dilate(img, kernel)
        return img

    def do(self, img):
        if self.noise and random.random() < 0.5:
            img = self.add_noise(img)
        if np.random.randint(0, 2):
            if self.dilate and random.random() < 0.5:
                img = self.add_dilate(img)
            elif self.erode and random.random() < 0.5:
                img = self.add_erode(img)
        return img


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
                draw.text((0, 0), self.lang_chars[i], (255, 255, 255),
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

    def __init__(self, width, height, need_crop):
        self.width = width
        self.height = height
        self.need_crop = need_crop

    def do(self, font_path, char, rotate=0):
        # 白色背景
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width))
        # 黑色字体
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        draw.text((x, y), char, (255, 255, 255), font=font)
        if np.random.randint(0, 2):
            for j in range(3):  # 条纹
                temp_x = np.random.randint(10, img.width - 10)
                temp_y = np.random.randint(10, img.height - 10)
                temp_a = np.random.choice((-1, 1))
                shape = [(temp_x, temp_y), (temp_x + 4 * temp_a, temp_y + 4 * temp_a)]
                draw.line(shape, fill="gray", width=0)
        #img.show()
        if rotate != 0:
            img = img.rotate(rotate)
        np_img = np.array(img)
        #print(np_img.shape)

        return np_img


def get_label_dict():
    filename = './digit.csv'
    label_dict = []
    with open(filename, 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            label_dict.append(row[0])  # 选择某一列加入到data数组中
    return label_dict


def create_img(char_list, font2image, verified_font_paths, rotate, test_ratio):
    test_digit = {}
    train_digit = {}
    data_aug = dataAugmentation()
    for i in range(11):  # 外层循环是字
        image_list = []
        for q in range(DATA_NUM):
            for j, verified_font_path in enumerate(verified_font_paths):  # 内层循环是字体
                k = np.random.randint(-rotate, rotate+1)
                image1 = font2image.do(verified_font_path, char_list[i], rotate=k)
                image2 = data_aug.do(image1)
                image_list.append(image2)

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
# if __name__ == '__main__':
    font_dir = os.path.expanduser('./digit_fonts')
    test_ratio = 0.2
    width = 44
    height = 44
    need_crop = False
    rotate = 10

    # 将汉字的label读入，得到（ID：汉字）的映射表label_dict
    char_list = get_label_dict()

    char_list = char_list[2:]
    # for i in range(len(char_list)):
    #     print(i, char_list[i])

    font_check = FontCheck(char_list)

    # 对于每类字体进行小批量测试
    verified_font_paths = []
    # search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)

    font2image = Font2Image(width, height, need_crop)

    train_digit, test_digit = create_img(char_list, font2image, verified_font_paths, rotate, test_ratio)

    # train_photos = np.array(list(train_digit.values()))
    # test_photos = np.array(list(test_digit.values()))
    # for i in range(train_photos.shape[0]):
    #     for j in range(train_photos.shape[1]):
    #         char_dir = os.path.join('./dataset/train', "%0.5d" % i)
    #
    #         if not os.path.isdir(char_dir):
    #             os.makedirs(char_dir)
    #         path_image = os.path.join(char_dir, "%d.png" % j)
    #         cv2.imwrite(path_image, train_photos[i][j])
    #     for j in range(test_photos.shape[1]):
    #         char_dir = os.path.join('./dataset/test', "%0.5d" % i)
    #
    #         if not os.path.isdir(char_dir):
    #             os.makedirs(char_dir)
    #         path_image = os.path.join(char_dir, "%d.png" % j)
    #         cv2.imwrite(path_image, test_photos[i][j])

    return train_digit, test_digit
