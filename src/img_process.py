# coding:utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import bmp_reader

def load_bmp(filename):
    tmp_file = './bmp_data.tmp'
    bmp_reader.readBMP(filename)
    if os.path.exists(tmp_file):
        bmp_arr = np.loadtxt(tmp_file).astype(np.int)
        os.remove(tmp_file)
        return bmp_arr
    else:
        raise Exception('Error with module bmp_reader!')

def show_greyval(img_arr):
    colum_sum = np.sum(img_arr,axis=0)
    row_sum = np.sum(img_arr,axis=1)

    plt.plot(colum_sum, label = 'horizontal')
    plt.plot(row_sum, color = 'red', label = 'vertical')
    plt.legend(loc='upper right')
    plt.xlabel('pixel_location')
    plt.ylabel('grey_value')
    plt.show()

def show_grey_histogram(img_arr):
    # print(np.histogram(img_arr))
    plt.hist(img_arr)
    plt.xlabel('grey_value')
    plt.ylabel('count')
    plt.show()

def pad_img(img_arr, rows, cols, scale):
    padded_img = np.pad(img_arr,((0,rows*(scale-1)),(0,cols*(scale-1))),'constant',constant_values = 0)
    return padded_img

def nearest_inter(img_arr, tar_height, tar_width):
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


def bilinear_inter(img_arr, tar_height, tar_width):
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

def S(x):
    x = np.abs(x)
    if 0 <= x < 1:
        return 1 - 2 * x * x + x * x * x
    if 1 <= x < 2:
        return 4 - 8 * x + 5 * x * x - x * x * x
    else:
        return 0

def trilinear_inter(img_arr,tar_height,tar_width):
    height,width =img_arr.shape
    new_img=np.zeros((tar_height,tar_width),np.uint8)
    sh=tar_height/height
    sw=tar_width/width
    for i in range(tar_height):
        for j in range(tar_width):
            x = i/sh
            y = j/sw
            p=(i+0.0)/sh-x
            q=(j+0.0)/sw-y
            x=int(x)-2
            y=int(y)-2
            if x>=1 and x<=(tar_height-3) and y>=1 and y<=(tar_width-3):
                A = np.array([
                    [S(1 + p), S(p), S(1 - p), S(2 - p)]
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
                    [S(1 + q)],
                    [S(q)],
                    [S(1 - q)],
                    [S(2 - q)]
                ])

                grey = np.dot(np.dot(A,B),C)

                if grey > 255:
                    grey = 255
                elif grey < 0:
                    grey = 0
                    
            else:
                grey = img_arr[x,y]

            new_img[i, j] = grey
 
 
    return new_img


# lena_grey = Image.open('./lena_pic/Lena.bmp').convert('L')
# img_arr = np.array(lena_grey)

img_arr = load_bmp('/Users/simon/Desktop/CV-Experiments/pic/Lena.bmp')
# print(img_arr)
[rows, cols] = img_arr.shape

# show_greyval(img_arr)
# show_grey_histogram(img_arr) # warning: slow

scale = 4

padded_arr = pad_img(img_arr, rows, cols, scale)
Image.fromarray(np.uint8(padded_arr)).save('../res/padded.png')

nearest_inter_arr = nearest_inter(img_arr, rows*scale, cols * scale)
Image.fromarray(np.uint8(nearest_inter_arr)).save('../res/nearest_inter.png')

bilinear_inter_arr = bilinear_inter(img_arr, rows*scale, cols * scale)
Image.fromarray(np.uint8(bilinear_inter_arr)).save('../res/bilinear_inter.png')

# trilinear_inter_arr = trilinear_inter(img_arr, rows*scale, cols * scale)
# Image.fromarray(np.uint8(trilinear_inter_arr)).save('../res/trilinear_inter.png')


