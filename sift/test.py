#%%
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from tqdm import tqdm

#%%
img = Image.open("raw.png").convert('L')
img_arr = np.array(img)


#%%
def get_scale_space(img, k, sigma, octave = 4, layer = 5, show = False):
    octaves = list()
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
            sigma = k*sigma
            layers.append(img_blur)

            if show:
                plt.subplot(octave,layer,i*layer+j+1)
                plt.imshow(img_blur,cmap='Greys_r')

        img = img.resize((int(h/2), int(w/2)))
        octaves.append(layers)

    if show:
        plt.show()

    return np.array(octaves)


#%%
def get_DoG(scale_space, show = False):
    octave, layer = scale_space.shape
    octaves = list()
    for i in range(octave):
        layers = list()
        for j in range(1, layer):
            diff = np.abs(scale_space[i][j].astype('int32')-scale_space[i][j-1].astype('int32'))
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

    return np.array(octaves)

#%%
def get_keypoints(DoG):
    octave, layer = DoG.shape
    offset = [-1, 0, 1]
    keypoints_alloct = list()

    total = 0
    for i in range(octave):
        for j in range(1, layer-1):
            diff = DoG[i][j]
            h, w = diff.shape
            # print(h, w)
            total += (h-2)*(w-2)

    with tqdm(total=total, ascii=True) as pbar:
        for i in range(octave):
            for j in range(1, layer-1):
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
                
                keypoints_alloct.append(keypoints)

    # total: octave * (layer-2) keypointmaps
    return keypoints_alloct

#%%

k = math.sqrt(2)
sigma = 1.6
scale_space = get_scale_space(img, k, sigma, 4, 5)
DoG = get_DoG(scale_space)
keypoints_alloct = get_keypoints(DoG)

#%%

keypoints = keypoints_alloct[0]
print('points num = ', len(keypoints))
plt.imshow(img, cmap=plt.cm.gray)
for keypoint in keypoints:
    plt.plot(keypoint[1], keypoint[0], 'y.')
plt.show()


#%%
