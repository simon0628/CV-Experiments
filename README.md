## 项目目标

**From Digital Image Processing & Computer Vision Experiments**

数据库：ImageNet，http://www.image-net.org/

1.下载至少2000张照片

2.图像大小更正，相同大小

3.基于直方图的图像增强

4.基于点操作的图像增强

5.基于领域操作的图像增强



## 工程结构

- src文件夹：存储程序源代码，主要为`img_process.py`文件
- `cv_environment.yaml`为Anaconda配置文件，用于移植实验环境
- res文件夹：存储程序运行结果
  - histogram子文件夹：存储各种处理结果的灰度直方图
  - res*.png：经过增强处理后的图像
  - temp子文件夹：经过大小更正后的图像
- pics文件夹：从ImageNet下载的5500张原图像



## 运行说明

1. `conda env create -f cv_environment.yaml`导入实验环境
1. `source activate image_process`进入实验环境
1. `python img_process.py`进行实验
1. 可以向程序传入相关设置参数，如`python img_process.py -number 100 -method 'average'`进行实验
   1. -number或-n为需要处理的图片数量，最大为5500，默认为20
   1. -method或-m为邻域处理的方法选择，包括average/max/mid，默认为average
1. 进入res文件夹查看实验结果



## 运行结果示例

##### 程序输出

```bash
> python3 img_process.py -n 5
开始进行图像处理，参数如下
    处理数量：5张
    点处理函数：对比度增强
    邻域处理函数：average滤波
处理中...
已处理1张
已处理2张
已处理3张
已处理4张
已处理5张
处理完毕!
```

##### 原图

![ILSVRC2017_test_00000001](doc/md_pic/ILSVRC2017_test_00000001.JPEG)



##### 灰度平衡结果

![res1_histo](doc/md_pic/res1_histo.png)

##### 点操作增强结果（增强对比度映射）

![res1_point](doc/md_pic/res1_point.png)

##### 邻域操作增强结果（均值滤波器）

![res1_area_average](doc/md_pic/res1_area_average.png)

##### 灰度直方图

![res1](doc/md_pic/res1.png)

## 使用的第三方库

numpy：用于数据科学处理

pillow：用于读取和写入图片

matplotlib：用于绘制数据统计图


