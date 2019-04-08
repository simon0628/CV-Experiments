# coding:utf-8

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# lena_grey = Image.open('./lena_pic/Lena.bmp').convert('L')

# lena_arr = np.array(lena_grey)

lena_arr = np.loadtxt('./lena_data.dat')
# print(lena_arr)
Image.fromarray(np.uint8(lena_arr)).show()

colum_sum = np.sum(lena_arr,axis=0)
row_sum = np.sum(lena_arr,axis=1)

# x = range(0,colum_sum.size)
plt.plot(colum_sum, label = 'horizontal')
plt.plot(row_sum, color = 'red', label = 'vertical')
plt.legend(loc='upper right')
plt.xlabel('pixel_location')
plt.ylabel('grey_value')
plt.show()
