# coding:utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os
# import cv2


def show_grey_histogram(img_arr):
    # print(np.histogram(img_arr))
    plt.hist(img_arr)
    plt.xlabel('grey_value')
    plt.ylabel('count')
    plt.show()


def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()


def equalHist(img):
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage

def pic_resize(temppath):
    # min_row = math.inf
    # min_col = math.inf
    for i in range(1,2000):
        readname = '../pics/ILSVRC2017_test_%08d.JPEG' % i
        writename = '%sILSVRC2017_test_%08d.JPEG' % (temppath, i)

        with Image.open(readname) as img:
            img.resize((500,333)).save(writename)
    #     [row, col] = Image.open(filename).size
    #     if row < min_row:
    #         min_row = row
    #     if col < min_col:
    #         min_col = col
    # print(min_row)
    # print(min_col)

pic_resize('../res/')
# lena_grey = Image.open('./pics/Lena.bmp').convert('L')
# img_arr = np.array(lena_grey)
# print(img_arr)
# [rows, cols] = img_arr.shape

# show_greyval(img_arr)
# show_grey_histogram(img_arr) # warning: slow

# scale = 4

# padded_arr = pad_img(img_arr, rows, cols, scale)
# Image.fromarray(np.uint8(padded_arr)).save('../res/padded.png')

# nearest_inter_arr = nearest_inter(img_arr, rows*scale, cols * scale)
# Image.fromarray(np.uint8(nearest_inter_arr)).save('../res/nearest_inter.png')

# bilinear_inter_arr = bilinear_inter(img_arr, rows*scale, cols * scale)
# Image.fromarray(np.uint8(bilinear_inter_arr)).save('../res/bilinear_inter.png')

# trilinear_inter_arr = trilinear_inter(img_arr, rows*scale, cols * scale)
# Image.fromarray(np.uint8(trilinear_inter_arr)).save('../res/trilinear_inter.png')

# show_FFT(img_arr)
# show_DCT(img_arr)
