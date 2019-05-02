# coding:utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os
# import cv2

def plot_grey_histogram(img_arr):
    hist = cal_grey_hist(img_arr)

    # max_index = np.argmax(hist)
    # hist[max_index] = 0
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

    point_pic_arr = point_enhance(pic_arr)
    p = plt.subplot(2,2,4)
    p.set_title('Histogram After Realm Enhance',fontsize='medium')
    plot_grey_histogram(point_pic_arr)

    plt.savefig(histpath + 'res' + str(cnt) + '.png')
    plt.close()

    Image.fromarray(np.uint8(equal_pic_arr)).save(respath + 'res' + str(cnt) + '_histo.png')
    Image.fromarray(np.uint8(point_pic_arr)).save(respath + 'res' + str(cnt) + '_point.png')
    cnt = cnt + 1
