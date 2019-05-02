# coding:utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os
# import cv2

def plot_grey_histogram(img_arr):
    hist = cal_grey_hist(img_arr)

    max_index = np.argmax(hist)
    hist[max_index] = hist[max_index-1]*1.1 if max_index > 0 else hist[max_index+1]*1.1

    # plt.xlim(0, 256)
    # plt.ylim(0, img_arr.max())
    # plt.axis([0, 255, 0, np.max(img_arr)])
    plt.bar(range(len(hist)),hist,width = 1, align = 'center',color='black', alpha = 0.8)

def cal_grey_hist(img_arr):
    hist = np.zeros([256], np.uint32)
    for row in img_arr:
        for pixel in row:
            # print(pixel)
            hist[pixel] = hist[pixel] + 1
            
    return hist

def equal_his(img_arr):
    # 灰度图像矩阵的高、宽
    h, w = img_arr.shape

    # 第一步：计算灰度直方图
    hist_arr = cal_grey_hist(img_arr)
 
    # 第二步：计算累加灰度直方图
    accumulation = np.zeros([256], np.uint32)
    accumulation[0] = hist_arr[0]
    for i in range(1,256):
        accumulation[i] = accumulation[i - 1] + hist_arr[i]
 
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    grey_cast = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for i in range(256):
        q = cofficient * float(accumulation[i]) - 1
        if q >= 0:
            grey_cast[i] = math.floor(q)
        else:
            grey_cast[i] = 0

    # 第四步：得到直方图均衡化后的图像
    hist_res = np.zeros(img_arr.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            hist_res[i][j] = grey_cast[img_arr[i][j]]
    return hist_res

def point_cast(x):
    if x <= 48:
        y = 0
    elif x > 218:
        y = 255
    else:
        y = 1.5*x-72

    if y > 255:
        y = 255
    return y

def point_enhance(img_arr):
    h, w = img_arr.shape
    res = np.zeros([h,w], np.uint8)
    for i in range(h):
        for j in range(w):
            res[i][j] = point_cast(img_arr[i][j])
    return res

def imgConvolve(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:卷积后的矩阵
    '''
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h+1, j - padding_w:j + padding_w+1]*kernel))

    return image_convolve


# 均值滤波
def imgAverageFilter(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:均值滤波后的矩阵
    '''
    return imgConvolve(image, kernel) * (1.0 / kernel.size)


# 高斯滤波
def imgGaussian(sigma):
    '''
    :param sigma: σ标准差
    :return: 高斯滤波器的模板
    '''
    img_h = img_w = 2 * sigma + 1
    gaussian_mat = np.zeros((img_h, img_w))
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            gaussian_mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return gaussian_mat


# Sobel Edge
def sobelEdge(image, sobel):
    '''
    :param image: 图片矩阵
    :param sobel: 滤波窗口
    :return:Sobel处理后的矩阵
    '''
    return imgConvolve(image, sobel)


# Prewitt Edge
def prewittEdge(image, prewitt_x, prewitt_y):
    '''
    :param image: 图片矩阵
    :param prewitt_x: 竖直方向
    :param prewitt_y:  水平方向
    :return:处理后的矩阵
    '''
    img_X = imgConvolve(image, prewitt_x)
    img_Y = imgConvolve(image, prewitt_y)

    img_prediction = np.zeros(img_X.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_X[i][j], img_Y[i][j])
    return img_prediction

# 滤波3x3
kernel_3x3 = np.ones((3, 3))
# 滤波5x5
kernel_5x5 = np.ones((5, 5))

# sobel 算子
sobel_1 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
# prewitt 算子
prewitt_1 = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_2 = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])

def area_enhance(img_arr, cast_size = 3, cast = [1,2]):
    h, w = img_arr.shape
    res = np.zeros([h,w], np.uint8)
    for i in range(h):
        for j in range(w):
            res[i][j] = point_cast(img_arr[i][j])
    return res

pic_num = 5

def pic_resize(temppath):
    # min_row = math.inf
    # min_col = math.inf
    for i in range(1, pic_num + 1):
        readname = '../pics/ILSVRC2017_test_%08d.JPEG' % i
        writename = '%sILSVRC2017_test_%08d.JPEG' % (temppath, i)

        with Image.open(readname) as img:
            img.convert('L').resize((500,333)).save(writename)


def read_pics(pic_path):
    pics_arr = list()
    for i in range(1, pic_num + 1):
        readname = pic_path + 'ILSVRC2017_test_%08d.JPEG' % i
        with Image.open(readname) as img:
            pics_arr.append(np.array(img, dtype = np.int32))
    return pics_arr



temppath = '../res/temp/'
respath = '../res/'
histpath = '../res/histogram/'

# pic_resize(temppath)

pics_arr = read_pics(temppath)

cnt = 1
for pic_arr in pics_arr:
    p = plt.subplot(2,2,1)
    p.set_title('Histogram of Raw Picture',fontsize='medium')
    plot_grey_histogram(pic_arr)
    

    equal_pic_arr = equal_his(pic_arr)
    p = plt.subplot(2,2,2)
    p.set_title('Histogram After Histo Equal',fontsize='medium')
    plot_grey_histogram(equal_pic_arr)

    point_pic_arr = point_enhance(pic_arr)
    p = plt.subplot(2,2,3)
    p.set_title('Histogram After Point Enhance',fontsize='medium')
    plot_grey_histogram(point_pic_arr)

    area_pic_arr = area_enhance(pic_arr)
    p = plt.subplot(2,2,4)
    p.set_title('Histogram After Area Enhance',fontsize='medium')
    plot_grey_histogram(area_pic_arr)

    plt.savefig(histpath + 'res' + str(cnt) + '.png')
    plt.close()

    Image.fromarray(np.uint8(equal_pic_arr)).save(respath + 'res' + str(cnt) + '_histo.png')
    Image.fromarray(np.uint8(point_pic_arr)).save(respath + 'res' + str(cnt) + '_point.png')
    Image.fromarray(np.uint8(area_pic_arr)).save(respath + 'res' + str(cnt) + '_area.png')
    cnt = cnt + 1
