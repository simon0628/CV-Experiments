#%%
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from tqdm import tqdm

#%%
img_raw = Image.open("small.png")
img = img_raw.convert('L')
img_arr = np.array(img)

def show_img(img):
    plt.imshow(img,cmap=plt.cm.gray)
    plt.show()
#%%
def get_scale_space(img, k, sigma, octave = 4, layer = 5, show = False):
    octaves = list()
    scales = list()
    new_sigma = sigma
    for i in range(octave):
        layers = list()
        h, w = img.size
        sigma = new_sigma

        for j in range(layer):
            if j == layer-3:
                new_sigma = sigma
            img_blur = cv2.GaussianBlur(np.array(img), (5,5), sigmaX = sigma)
            # img_blur = np.array(img_blur, dtype='int8')
            scales.append(sigma)
            layers.append(img_blur)
            sigma = k*sigma

            if show:
                plt.subplot(octave,layer,i*layer+j+1)
                plt.imshow(img_blur,cmap='Greys_r')

        img = img.resize((int(h/2), int(w/2)))
        octaves.append(layers)

    if show:
        plt.show()

    return np.array(octaves), scales


#%%
def get_DoG(scale_space, scales, show = False):
    octave, layer = scale_space.shape
    new_scales = list()
    octaves = list()
    scale_cnt = 0
    for i in range(octave):
        layers = list()
        scale_cnt += 1
        for j in range(1, layer):
            diff = np.abs(scale_space[i][j].astype('int32')-scale_space[i][j-1].astype('int32'))
            new_scales.append(scales[scale_cnt])
            scale_cnt += 1
            # print(diff)
            # plt.imshow(diff)
            # plt.show()
            layers.append(diff.astype('uint8'))

            if show:
                plt.subplot(octave,layer-1,i*(layer-1)+j)
                plt.imshow(diff, cmap=plt.cm.gray)
        
        octaves.append(layers)

    if show:
        plt.show()

    return np.array(octaves), new_scales

#%%
def get_keypoints(DoG, scales):
    octave, layer = DoG.shape
    offset = [-1, 0, 1]
    keypoints_octs = list()
    scale_cnt = 0
    new_scales = list()

    total = 0
    for i in range(octave):
        for j in range(1, layer-1):
            diff = DoG[i][j]
            h, w = diff.shape
            # print(h, w)
            total += (h-2)*(w-2)

    with tqdm(total=total, ascii=True) as pbar:
        for i in range(octave):
            scale_cnt += 1
            for j in range(1, layer-1):
                new_scales.append(scales[scale_cnt])
                scale_cnt += 1
                keypoints = list()
                diff = DoG[i][j]
                h, w = diff.shape
                for ii in range(1, h-1):
                    for jj in range(1, w-1):
                        ismax = True
                        ismin = True
                        # print('p   :', diff[ii][jj])
                        # 3*3*3-1 = 26
                        for d1 in offset:
                            if not ismax and not ismin:
                                break
                            for d2 in offset:
                                if not ismax and not ismin:
                                    break
                                for d3 in offset:
                                    if not ismax and not ismin:
                                        break
                                    # print('near:', DoG[i][j+d1][ii+d2][jj+d3])
                                    if d1 == 0 and d2 == 0 and d3 == 0:
                                        continue
                                    if ismax and diff[ii][jj] <= DoG[i][j+d1][ii+d2][jj+d3]:
                                        ismax = False
                                    if ismin and diff[ii][jj] >= DoG[i][j+d1][ii+d2][jj+d3]:
                                        ismin = False
                        # print('ismax :', ismax, ' ismin :', ismin)
                        if ismax or ismin:
                            keypoints.append([ii, jj])
                        pbar.update(1)
                keypoints_octs.append(keypoints)
            scale_cnt += 1

    # total: octave * (layer-2) keypointmaps
    return keypoints_octs, new_scales

#%%

k = math.sqrt(2)
sigma = 1.6
scale_space, scales = get_scale_space(img, k, sigma, 4, 5)
DoG, DoG_scales = get_DoG(scale_space, scales)
keypoints_octs, kp_scales = get_keypoints(DoG, DoG_scales)

#%%

keypoints = keypoints_octs[0]
print('points num = ', len(keypoints))
plt.imshow(img_raw, cmap=plt.cm.gray)
for keypoint in keypoints:
    plt.plot(keypoint[1], keypoint[0], 'r.')
plt.show()

#tylor expandsion and hessian elimination

def padding(img, size):
    h,w = img.shape

    convolve_h = int(h + 2 * size)
    convolve_w = int(w + 2 * size)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_w))
    # 中心填充图片
    img_padding[size:size + h, size:size + w] = img[:, :]
    return img_padding

#%%
def cal_grad(img, sigma):

    img_padding = padding(img, 1)

    m = np.zeros((h,w))
    theta = np.zeros((h,w))

    for i in range(h):
        for j in range(w):
            m[i][j] = np.sqrt(np.square(img_padding[i+2][j] - img_padding[i][j])+np.square(img_padding[i][j+2] - img_padding[i][j]))
            if img_padding[i][j+2] - img_padding[i][j] == 0:
                if img_padding[i+2][j] - img_padding[i][j] > 0:
                    theta[i][j] = math.pi / 2
                elif img_padding[i+2][j] - img_padding[i][j] < 0:
                    theta[i][j] = -math.pi/2
                else:
                    theta[i][j] = 0
            else:
                theta[i][j] = np.arctan((img_padding[i+2][j] - img_padding[i][j])/(img_padding[i][j+2] - img_padding[i][j]))
    
    img_blur = cv2.GaussianBlur(m, (5,5), sigmaX = 1.5*sigma)

    print(m)
    print(theta)
    return m, theta

def get_ori(DoG, DoG_scales, keypoints_octs, kp_scales):
    octave, level = DoG.shape
    DoG_scale_cnt = 0
    ori_padding = 10
    ori_num = 360.ori_padding

    for i in range(octave):
        DoG_scale_cnt += 1
        for j in range(1, level-1):
            m, theta = cal_grad(DoG[i][j], DoG_scales[DoG_scale_cnt])
            radius = int(DoG_scales[DoG_scale_cnt] * 1.5 * 3)
            DoG_scale_cnt += 1
            
            DoG_padding = padding(DoG, radius)
            keypoints = keypoints_octs[i][j-1]
            h,w = m.shape
            for keypoint in keypoints:
                ori_histogram = np.zeros(ori_num)
                for i_offset in range(-radius, radius):
                    for j_offset in range(-radius, radius):
                        i_sub = i+i_offset
                        j_sub = j+j_offset
                        if i_sub > 0 and i_sub < h and j_sub > 0 and j_sub < w:
                            ind = int((theta[i_sub][j_sub] + pi/2) * 180/pi / ori_num)
                            ori_histogram[ind] += m[i_sub][j_sub]


                

        DoG_scale_cnt += 1


m, theta = cal_grad(img_arr, DoG_scales[0])
show_img(m)

#%%
