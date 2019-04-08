# coding:utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# lena_grey = Image.open('./lena_pic/Lena.bmp').convert('L')

# lena_arr = np.array(lena_grey)

lena_arr = np.loadtxt('./lena_data.dat').astype(np.int)
# print(lena_arr)

scale = 4

padded_arr = np.pad(lena_arr,((0,lena_arr.shape[0]*3),(0,lena_arr.shape[0]*3)),'constant',constant_values = 0)
# print(padded_arr.shape)
# print(padded_arr)
Image.fromarray(np.uint8(padded_arr)).show()

# repeat_arr = np.repeat(np.repeat(lena_arr, scale, axis = 0),scale,axis = 1)
# print(repeat_arr.shape)
# print(repeat_arr)
# Image.fromarray(np.uint8(repeat_arr)).show()


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


[rows, cols] = lena_arr.shape
nearest_inter_arr = nearest_inter(lena_arr, rows*scale, cols * scale)
Image.fromarray(np.uint8(nearest_inter_arr)).show()

bilinear_inter_arr = bilinear_inter(lena_arr, rows*scale, cols * scale)
Image.fromarray(np.uint8(bilinear_inter_arr)).show()


# colum_sum = np.sum(lena_arr,axis=0)
# row_sum = np.sum(lena_arr,axis=1)

# plt.plot(colum_sum, label = 'horizontal')
# plt.plot(row_sum, color = 'red', label = 'vertical')
# plt.legend(loc='upper right')
# plt.xlabel('pixel_location')
# plt.ylabel('grey_value')
# plt.show()
