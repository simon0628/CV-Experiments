# coding:utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os
import sys

class img_resize():

    def nearest_inter(self, img_arr, tar_height, tar_width):
        [height, width] = img_arr.shape
        new_img = np.zeros((tar_height, tar_width), np.uint8)
        sh = tar_height/height
        sw = tar_width/width
        for i in range(tar_height):
            for j in range(tar_width):
                x = int(i/sh)
                y = int(j/sw)
                new_img[i, j] = img_arr[x, y]
        return new_img


    def bilinear_inter(self,img_arr, tar_height, tar_width):
        [height, width] = img_arr.shape
        new_img = np.zeros((tar_height, tar_width), np.uint8)
        sh = tar_height/height
        sw = tar_width/width
        for i in range(tar_height):
            for j in range(tar_width):
                x = i/sh
                y = j/sw
                p = (i+0.0)/sh-x
                q = (j+0.0)/sw-y
                x = int(x)-1
                y = int(y)-1
                if x+1 < tar_height and y+1 < tar_width:
                    new_img[i, j] = int(img_arr[x][y]*(1-p)*(1-q)+img_arr[x][y+1]
                                        * q*(1-p)+img_arr[x+1][y]*(1-q)*p+img_arr[x+1][y+1]*p*q)
        return new_img


    def S(self, x):
        x = np.abs(x)
        if 0 <= x < 1:
            return 1 - 2 * x * x + x * x * x
        if 1 <= x < 2:
            return 4 - 8 * x + 5 * x * x - x * x * x
        else:
            return 0


    def trilinear_inter(self,img_arr, tar_height, tar_width):
        height, width = img_arr.shape
        new_img = np.zeros((tar_height, tar_width), np.uint8)
        sh = tar_height/height
        sw = tar_width/width
        for i in range(tar_height):
            for j in range(tar_width):
                x = i/sh
                y = j/sw
                p = (i+0.0)/sh-x
                q = (j+0.0)/sw-y
                x = int(x)-2
                y = int(y)-2
                if x >= 1 and x <= (tar_height-3) and y >= 1 and y <= (tar_width-3):
                    A = np.array([
                        [self.S(1 + p), self.S(p), self.S(1 - p), self.S(2 - p)]
                    ])

                    B = np.array([
                        [img_arr[x-1, y-1], img_arr[x-1, y],
                        img_arr[x-1, y+1],
                        img_arr[x-1, y+1]],
                        [img_arr[x, y-1], img_arr[x, y],
                        img_arr[x, y+1], img_arr[x, y+2]],
                        [img_arr[x+1, y-1], img_arr[x+1, y],
                        img_arr[x+1, y+1], img_arr[x+1, y+2]],
                        [img_arr[x+2, y-1], img_arr[x+2, y],
                        img_arr[x+2, y+1], img_arr[x+2, y+1]],

                    ])

                    C = np.array([
                        [self.S(1 + q)],
                        [self.S(q)],
                        [self.S(1 - q)],
                        [self.S(2 - q)]
                    ])

                    grey = np.dot(np.dot(A, B), C)

                    if grey > 255:
                        grey = 255
                    elif grey < 0:
                        grey = 0

                else:
                    grey = img_arr[x, y]

                new_img[i, j] = grey

        return new_img


# 绘制图像灰度直方图
def plot_grey_histogram(img_arr):
    '''
    :param image_arr: 图片矩阵
    '''
    hist = cal_grey_hist(img_arr)

    max_index = np.argmax(hist)
    hist[max_index] = hist[max_index-1]*1.1 if max_index > 0 else hist[max_index+1]*1.1

    # 将直方图拉伸至plot边缘
    # plt.xlim(0, 256)
    # plt.ylim(0, img_arr.max())
    # plt.axis([0, 255, 0, np.max(img_arr)])
    plt.bar(range(len(hist)),hist,width = 1, align = 'center',color='black', alpha = 0.8)


def cal_grey_hist(img_arr):
    '''
    :param image_arr: 图片矩阵
    :return: hist: 256维数组，存储每种灰度值的统计数量
    '''
    hist = np.zeros([256], np.uint32)
    for row in img_arr:
        for pixel in row:
            # print(pixel)
            hist[pixel] = hist[pixel] + 1
            
    return hist

# 直方图均衡
def equal_his(img_arr):
    '''
    :param image_arr: 图片矩阵
    :return: hist_res: 图像经过直方图均衡化的结果
    '''    
    h, w = img_arr.shape

    # 计算灰度值统计量
    hist_arr = cal_grey_hist(img_arr)
 
    # 计算灰度值的累加数组
    accumulation = np.zeros([256], np.uint32)
    accumulation[0] = hist_arr[0]
    for i in range(1,256):
        accumulation[i] = accumulation[i - 1] + hist_arr[i]
 
    # 根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    grey_cast = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for i in range(256):
        q = cofficient * float(accumulation[i]) - 1
        if q >= 0:
            grey_cast[i] = math.floor(q)
        else:
            grey_cast[i] = 0

    # 得到直方图均衡化后的图像
    hist_res = np.zeros(img_arr.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            hist_res[i][j] = grey_cast[img_arr[i][j]]
    return hist_res


def init_gammar_cast(c, y):
    gammar_cast_arr = np.zeros(256)
    max = c * math.pow(255, y)
    for x in range(255):
        gammar_cast_arr[x] = c * math.pow(x, y) / max * 256
    return gammar_cast_arr

def point_cast(x, method):
    '''
    :param x: 输入灰度值
    :return: y: 映射结果
    :note: 点操作的映射函数    
    '''

    if method == 'Z-improve-contrast':
        # 采用Z型函数，可提高对比度
        if x <= 48:
            y = 0
        elif x > 218:
            y = 255
        else:
            y = 1.5*x-72
    
    elif method == 'linear':
        # 降低对比度
        y = 0.56 * x
    
    elif method == 'log':
        # 低灰度区扩展，高灰度区压缩
        # y = 32*log(1+x)
        # x = 0 -> y = 0
        # x = 255 -> y = 255
        y = 32 * math.log2(1+x)

    # 规范输出灰度值范围
    if y > 255:
        y = 255
    if y < 0:
        y = 0
    return y

# 基于点处理的增强
def point_enhance(img_arr, method = 'linear'):
    '''
    :param image_arr: 图片矩阵
    :return: res: 图像经过点增强的运算结果
    '''    
    if method == 'gammar':
        y = 1/2.2
        c = 255/(math.pow(255, 0.45))
        gammar_cast = init_gammar_cast(c, y)
    h, w = img_arr.shape
    res = np.zeros([h,w], np.uint8)
    for i in range(h):
        for j in range(w):
            if method == 'gammar':
                res[i][j] = gammar_cast[img_arr[i][j]]
            else:
                res[i][j] = point_cast(img_arr[i][j], method)
    return res

# 图片卷积操作
def img_convolve(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel_size: 滤波窗口size
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
    convolve_w = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_w))
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
def average_filter(image, kernel_size):
    '''
    :param image: 图片矩阵
    :param kernel_size: 滤波窗口size
    :return:均值滤波后的矩阵
    '''
    kernel = np.ones((kernel_size, kernel_size))
    return img_convolve(image, kernel) * (1.0 / kernel.size)

# 最大值滤波
def max_filter(image, kernel_size, max_min = False):
    '''
    :param image: 图片矩阵
    :param kernel_size: 滤波窗口size
    :return:最大值滤波后的矩阵
    '''

    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    # padding
    padding = int((kernel_size - 1) / 2)

    padding_h = int(img_h + 2 * padding)
    padding_w = int(img_w + 2 * padding)

    # 分配空间
    img_padding = np.zeros((padding_h, padding_w))
    # print(img_padding.shape)
    # 中心填充图片
    img_padding[padding:padding + img_h, padding:padding + img_w] = image[:, :]

    res = np.zeros((img_h, img_w))

    for i in range(padding, padding + img_h):
        for j in range(padding, padding + img_w):
            filtered = img_padding[i-padding:i+padding+1, j-padding:j+padding+1]
            if max_min == False:
                res[i-padding][j-padding] = max(map(max,filtered))      
            else:
                res[i-padding][j-padding] = min(map(min,filtered))      
    return res

# 中值滤波
def mid_filter(image, kernel_size):
    '''
    :param image: 图片矩阵
    :param kernel_size: 滤波窗口size
    :return:中值滤波后的矩阵
    '''
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    # padding
    padding = int((kernel_size - 1) / 2)

    padding_h = int(img_h + 2 * padding)
    padding_w = int(img_w + 2 * padding)

    # 分配空间
    img_padding = np.zeros((padding_h, padding_w))
    # print(img_padding.shape)
    # 中心填充图片
    img_padding[padding:padding + img_h, padding:padding + img_w] = image[:, :]

    res = np.zeros((img_h, img_w))

    for i in range(padding, padding + img_h):
        for j in range(padding, padding + img_w):
            filtered = img_padding[i-padding:i+padding+1, j-padding:j+padding+1]
            res[i-padding][j-padding] = np.median(filtered)    
    return res


# 高斯滤波
def gaussian(sigma):
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
def sobel_filter(image, sobel):
    '''
    :param image: 图片矩阵
    :param sobel: 滤波窗口
    :return:Sobel处理后的矩阵
    '''
    return img_convolve(image, sobel)


# Prewitt Edge
def prewitt_filter(image, prewitt_x, prewitt_y):
    '''
    :param image: 图片矩阵
    :param prewitt_x: 竖直方向
    :param prewitt_y:  水平方向
    :return:处理后的矩阵
    '''
    img_X = img_convolve(image, prewitt_x)
    img_Y = img_convolve(image, prewitt_y)

    img_prediction = np.zeros(img_X.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_X[i][j], img_Y[i][j])
    return img_prediction


# 基于邻域处理的增强
def area_enhance(img_arr, method=None):
    '''
    :param image_arr: 图片矩阵
    :return: res: 图像经过邻域增强的运算结果
    '''   
    h, w = img_arr.shape

    if method == 'sobel':
        res = np.array(sobel_filter(img_arr, sobel_1), dtype=np.int)
    elif method == 'prewitt':
        res = np.array(prewitt_filter(img_arr, prewitt_1, prewitt_2), dtype=np.int)
    elif method == 'max':
        res = np.array(max_filter(img_arr, 3), dtype=np.int)
    elif method == 'min':
        res = np.array(max_filter(img_arr, 3, True), dtype=np.int)
    elif method == 'mid':
        res = np.array(mid_filter(img_arr, 3), dtype=np.int)
    else:
        res = np.array(average_filter(img_arr, 3), dtype=np.int)

    h, w = res.shape
    for i in range(h):
        for j in range(w):
            if res[i][j] > 255:
                res[i][j] = 255
            elif res[i][j] < 0:
                res[i][j] = 0
    return res

# 为图片添加椒盐噪声
def addsalt_pepper(img, SNR = 0.9):
    res = img.copy()
    h, w = res.shape
    mask = np.random.choice((0, 1, 2), size=(h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    res[mask == 1] = 255    # 盐噪声
    res[mask == 2] = 0      # 椒噪声
    return res

# 图像大小更正，统一为500*333，并存储在临时路径中
def pic_resize(temppath):
    for i in range(1, pic_num + 1):
        readname = '../pic/ILSVRC2017_test_%08d.JPEG' % i
        writename = '%sILSVRC2017_test_%08d.JPEG' % (temppath, i)

        with Image.open(readname) as img:
            # plt.plot(img.size[0], img.size[1], 'k*') # 统计所有图片的尺寸
            img.convert('L').resize((500,333)).save(writename)

# 从目标路径中读取图片，需要保证图片已经预处理完毕，直接可用
def read_pics(pic_path):
    pics_arr = list()
    for i in range(1, pic_num + 1):
        readname = pic_path + 'ILSVRC2017_test_%08d.JPEG' % i
        with Image.open(readname) as img:
            pics_arr.append(np.array(img, dtype = np.int32))
    return pics_arr


# 工程路径
temppath = '../res/temp/'
respath = '../res/'
histpath = '../res/histogram/'


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


pic_num = 3
area_filter_method = 'average'
point_filter_method = 'linear'

for i in range(len(sys.argv)):
    if sys.argv[i] == '-number' or sys.argv[i] ==  '-n':
        pic_num = int(sys.argv[i+1])
    if sys.argv[i] == '-area' or sys.argv[i] ==  '-a':
        area_filter_method = sys.argv[i+1]
    if sys.argv[i] == '-point' or sys.argv[i] ==  '-p':
        point_filter_method = sys.argv[i+1]

print('开始进行图像处理，参数如下')
print('    处理数量：%d张' % pic_num)
print('    点处理函数：%s函数' % point_filter_method)
print('    邻域处理函数：%s滤波' % area_filter_method)
print('处理中...')



# pic_resize(temppath)

pics_arr = read_pics(temppath)

cnt = 1
for pic_arr in pics_arr:
    # # 绘制原始图像灰度直方图
    # p = plt.subplot(2,2,1)
    # p.set_title('Histogram of Raw Picture',fontsize='medium')
    # plot_grey_histogram(pic_arr)
    
    # # 直方图均衡
    # equal_pic_arr = equal_his(pic_arr)
    # p = plt.subplot(2,2,2)
    # p.set_title('Histogram After Histo Equal',fontsize='medium')
    # plot_grey_histogram(equal_pic_arr)
    # Image.fromarray(np.uint8(equal_pic_arr)).save(respath + 'res' + str(cnt) + '_histo.png')

    # # 点操作的图像增强
    # point_pic_arr = point_enhance(pic_arr, point_filter_method)
    # p = plt.subplot(2,2,3)
    # p.set_title('Histogram After Point Enhance',fontsize='medium')
    # plot_grey_histogram(point_pic_arr)
    # Image.fromarray(np.uint8(point_pic_arr)).save(respath + 'res' + str(cnt) + '_point_' + point_filter_method + '.png')

    pic_arr = addsalt_pepper(pic_arr, 0.95)
    Image.fromarray(np.uint8(pic_arr)).save(respath + 'res' + str(cnt) + '_pepper.png')

    # 邻域操作的图像增强
    area_pic_arr = area_enhance(pic_arr, area_filter_method)
    p = plt.subplot(2,2,4)
    p.set_title('Histogram After Area Enhance',fontsize='medium')
    plot_grey_histogram(area_pic_arr)
    Image.fromarray(np.uint8(area_pic_arr)).save(respath + 'res' + str(cnt) + '_area_' + area_filter_method + '.png')


    # Sobel和prewitt边缘处理
    # area_filter_method = 'sobel'
    # area_pic_arr = area_enhance(pic_arr, area_filter_method)
    # Image.fromarray(np.uint8(area_pic_arr)).save(respath + 'res' + str(cnt) + '_area_' + area_filter_method + '.png')

    # area_filter_method = 'prewitt'
    # area_pic_arr = area_enhance(pic_arr, area_filter_method)
    # Image.fromarray(np.uint8(area_pic_arr)).save(respath + 'res' + str(cnt) + '_area_' + area_filter_method + '.png')

    # plt.tight_layout()

    # plt.savefig(histpath + 'res' + str(cnt) + '.png')
    # plt.close()

    print('已处理%d张' % cnt)
    cnt = cnt + 1
print('处理完毕!')
