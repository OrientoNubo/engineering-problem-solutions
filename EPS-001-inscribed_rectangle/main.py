#!/usr/bin/env python
# _*_ coding:utf-8 _*_
#
# @Version : 1.0
# @Time    : 2023/01/10
# @Author  : Si-yu, Lu
# @File    : main.py


import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from numba import jit
import math
from rich.progress import track
import time

'''
说明：
    这里使用了opencv内建的一些functions，这些functions也可以手磕。
'''


@jit(nopython=True)
def Dilation(img, kernel=None):
    h, w = img.shape
    new_img = np.zeros(shape=img.shape, dtype=np.uint8)
    # pixel_skip = np.floor(kernel.shape[0] / 2).astype(int)
    pixel_skip = 2
    
    # pixel loop
    for i in range(pixel_skip, h - pixel_skip):
        for j in range(pixel_skip, w - pixel_skip):
            # kernel loop
            temp = []
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    # skip useless pixel 
                    if kernel[m][n] == 0:
                        continue
                    temp.append(img[i + m - pixel_skip][j + n - pixel_skip])

            new_img[i][j] = min(temp)
    return new_img

def Manual_INTER_LINEAR(img, sfactor):
    org_h, org_w, channel = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
    new_h, new_w = int(img.shape[0] * sfactor), int(img.shape[1] * sfactor)
    img_resized = np.zeros((new_h, new_w, channel), dtype=np.uint8)

    for h in range(new_h):
        for w in range(new_w):
            org_x, org_y = float((w + 0.5) * (org_w / new_w) - 0.5), float((h + 0.5) * (org_h / new_h) - 0.5)
            org_x_int, org_y_int = math.floor(org_x), math.floor(org_y)
            org_x_float, org_y_float = org_x - org_x_int, org_y - org_y_int

            if org_x_int + 1 == org_w or org_y_int + 1 == org_h:
                img_resized[h, w, :] = img[org_y_int, org_x_int, :]
                continue

            img_resized[h][w][:] =  (1. - org_y_float) * (1. - org_x_float) * img[org_y_int, org_x_int, :] + \
                                    (1. - org_y_float) * org_x_float * img[org_y_int, org_x_int + 1, :] + \
                                    org_y_float * (1. - org_x_float) * img[org_y_int + 1, org_x_int, :] + \
                                    org_y_float * org_x_float * img[org_y_int + 1, org_x_int + 1, :]
    return img_resized

def Count_Valid_Pixels(img):
        _h, _w = img.shape
        count = 0
        ch, cw = 0, 0
        for h in range(_h):
            for w in range(_w):
                if img[h, w] == 255:
                    count += 1
        if count == 1:
            for h in range(_h):
                for w in range(_w):
                    if img[h, w] == 255:
                        ch, cw = h, w
        return count, [cw, ch]

@jit(nopython=True)
def Get_Neighbor_By_KernelSize(img: np.array, pos: tuple, size: tuple) -> np.array:
    '''
    将测试图像中目前像素索引的邻域提取出来，这个邻域的尺寸取决于Kernel的大小
    '''
    Neighbor = np.zeros(shape=(size[0], size[1]), dtype=np.uint8)
    for pos_h in range(size[0]):
        for pos_w in range(size[1]):
            pos_nh, pos_nw = pos[0] + pos_h, pos[1] + pos_w
            if (pos_nh < 0 or pos_nh >= img.shape[0]) or (pos_nw < 0 or pos_nw >= img.shape[1]):
                return np.zeros(shape=(size[0], size[1]), dtype=np.uint8)
            Neighbor[pos_h][pos_w] = img[pos_nh][pos_nw]
    return Neighbor

@jit(nopython=True)
def Matching_Handle(img, kernel):
    # 以kernel的左上角作为起点，从左到右、从上到下，一步一步映射到测试图像
    Matching_list = []
    k_h, k_w = kernel.shape
    print("kernel.shape:", k_h, k_w)
    
    # 遍历测试图像的每个像素
    for h in range(_h):
        for w in range(_w):
            
            # 在测试图像中提取要匹配的邻域
            Neighbor = Get_Neighbor_By_KernelSize(img, [h, w], [k_h, k_w])
            
            # 判断取出的邻域像素是否能与Kernel匹配
            if Neighbor.all() != None:
                CannotMatchFlag = False
                for i in range(k_h):
                    for j in range(k_w):
                        if kernel[i][j] == 1:
                            if Neighbor[i][j] == 255:
                                pass
                            else:
                                CannotMatchFlag = True
                        else: pass
            else: continue
            
            if CannotMatchFlag != True:
                Matching_list.append([h, w])
                
    return Matching_list


if __name__ == '__main__':
    time_start = time.time()
    
    ### ================== setting ================== ###
    
    img_path = './test.png'             # 输入图像路径
    results_root = './output/'          # 输出图像路径
    region_color_high = [255, 12, 12]       # RBG Int Value
    region_color_low  = [245, 0, 0]         # RGB Int Value
    scale_factor = 0.125                      # Float Value 测试图像的缩放比例
    aspect_ratio_list = [(1/3), (1/2.5), (1/2), (1/1.5), 1, 1.5, 2, 2.5, 3]     # 矩形的长宽比设定，初始长度不用设定（让它暴力尝试）
    angle_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]    # 只要45度即可
    
    
    
    ### ================== handle ================== ###
    
    org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)        # 读取平面的像素（输入彩色）
    org_h, org_w = org_img[:, :, 0].shape
    
    if org_w > org_h:
        padding = int((org_w - org_h) / 2)
        img_padding = np.zeros(shape=(org_w, org_w, 3), dtype=np.uint8)
        img_padding[padding : padding + org_h, :, :] = org_img

    else:
        padding = int((org_h - org_w) / 2)
        img_padding = np.zeros(shape=(org_h, org_h, 3), dtype=np.uint8)
        img_padding[padding : padding + org_w, :, :] = org_img
        
        
    center = (int(org_w // 2), int(org_w // 2))
    
    
    for angle in track(angle_list):
    
        # 开始旋转原始图像
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotate_padding = cv2.warpAffine(img_padding, M, (img_padding.shape[0], img_padding.shape[1]))
        
        # org_img = img_padding
        # cv2.imwrite(results_root + 'rotate_padding_' + str(angle) + '.png', rotate_padding)
        
        org_img = rotate_padding
        
        org_img = Manual_INTER_LINEAR(org_img, scale_factor)    # 缩放一下，减少计算量
        _h, _w = org_img[:, :, 0].shape       # 读取图像的像素尺寸
        binarized_image = np.zeros(shape=org_img[:,:,0].shape, dtype=np.uint8)         # 新建img以存储经过二值化判断的图像
        
    
        
        # 利用设定里的颜色范围进行二值化操作
        for h in range(_h):
            for w in range(_w):
                # 如果符合区域颜色的范围，就标记为【255白色】，注意python-opencv的彩色通道是按照BGR排列的
                if org_img[h][w][2] >= region_color_low[0] and org_img[h][w][2] <= region_color_high[0] and org_img[h][w][1] >= region_color_low[1] and org_img[h][w][1] <= region_color_high[1] and org_img[h][w][0] >= region_color_low[2] and org_img[h][w][0] <= region_color_high[2]:
                    binarized_image[h][w] = 255
                # 如果不符合区域颜色的范围，泽标记为【0黑色】
                else:
                    binarized_image[h][w] = 0
        
        # cv2.imwrite(results_root + 'output_1.png', binarized_image)    # 存储二值化图像
        print(binarized_image.shape)


        rotate_img_startpoint_list = []
        rotate_img_kernel_list = []
        rotate_img_acreage_list = []
        
        
        output_rectangle = binarized_image.copy()
        
        # 使用模板暴力匹配操作(找面积最大矩形)
        # 模板须是一个矩形的矩形的kernel
        if True:
            # 这里开始，处理单张测试图片
            
            
            for aspect_ratio in aspect_ratio_list:
                print("########################### ", aspect_ratio, " ###########################")
                
                if binarized_image.shape[0] > binarized_image.shape[1]: 
                    kernel_size_h_init = binarized_image.shape[0] / 4
                else: 
                    kernel_size_h_init = binarized_image.shape[1] / 4

                kernel_size_h, kernel_size_w = int(kernel_size_h_init), int((kernel_size_h_init)*aspect_ratio)
                kernel = np.ones([kernel_size_h, kernel_size_w], dtype='uint8')
                kernel_size_h_list = []
                
                for iter in range(100):
                    print("============", "iter num:", iter, "============")
                    
                    Matching_list = Matching_Handle(binarized_image, kernel)
                    print("Matching_list len:", np.shape(Matching_list))
                    
                    # 判断处理“无法减少到唯一解”的情况，这个操作可以加到下面的if结构里
                    print("kernel_size_h_list: ", kernel_size_h_list)
                    print("kernel_size_h: ", kernel_size_h)
                    if np.shape(Matching_list)[0] > 1:
                        if kernel_size_h in kernel_size_h_list:
                            
                            for i in range(np.shape(Matching_list)[0]):
                                
                                # print(Matching_list[i])
                                # 记录到全局list
                                rotate_img_startpoint_list.append(Matching_list[i])
                                rotate_img_kernel_list.append([kernel_size_h, kernel_size_w])
                                rotate_img_acreage_list.append(kernel_size_h*kernel_size_w)
                                
                            break
                
                    # 判断目前Kernel的个数，逻辑为：如果目前匹配数为0，那么缩小Kernel；如果目前匹配数大于1，那么放大Kernel
                    if np.shape(Matching_list)[0] == 0:
                        print("No matching one!")
                        kernel_size_h, kernel_size_w = int(kernel_size_h-1), int((kernel_size_h-1)*aspect_ratio)
                        kernel = np.ones([kernel_size_h, kernel_size_w], dtype='uint8')
                        
                    elif np.shape(Matching_list)[0] > 1:
                        print("Still too many matching!")
                        kernel_size_h_list.append(kernel_size_h)
                        kernel_size_h, kernel_size_w = int(kernel_size_h+1), int((kernel_size_h+1)*aspect_ratio)
                        kernel = np.ones([kernel_size_h, kernel_size_w], dtype='uint8')

                    elif np.shape(Matching_list)[0] == 1:
                        print("Found the best matching!")
                        print(Matching_list)
                        
                        # 记录到全局list
                        rotate_img_startpoint_list.append(Matching_list[0])
                        rotate_img_kernel_list.append([kernel_size_h, kernel_size_w])
                        rotate_img_acreage_list.append(kernel_size_h*kernel_size_w)
                        
                        break
                        
                    else:
                        print("Error Case.")
    
        print("rotate_img_startpoint_list :: ", rotate_img_startpoint_list)      
        print("rotate_img_kernel_list     :: ", rotate_img_kernel_list)      
        print("rotate_img_acreage_list    :: ", rotate_img_acreage_list)

        # 取面积最大的输出
        acreage_max = max(rotate_img_acreage_list)
        for i in range(len(rotate_img_acreage_list)):
            if rotate_img_acreage_list[i] == acreage_max:
                
                output_rectangle = cv2.rectangle(output_rectangle, (rotate_img_startpoint_list[i][1], rotate_img_startpoint_list[i][0]), (rotate_img_startpoint_list[i][1] + rotate_img_kernel_list[i][1], rotate_img_startpoint_list[i][0] + rotate_img_kernel_list[i][0]), [125])
                cv2.imwrite(results_root + 'output_rectangle_size-' + str(acreage_max) + '_angle-' + str(angle) + '.png', output_rectangle)
            
        
        
 
    # 画最大圆的一种方式，使用膨胀操作(找面积最大的圆)
    if False:
        dilation_image = binarized_image
        kernel = np.array([
                            [0,1,1,1,0],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [0,1,1,1,0]
                        ], dtype='uint8')
        dist = 0 
        count = 0
        while(count != 1):
            dilation_image = Dilation(dilation_image, kernel)
            count, _ = Count_Valid_Pixels(dilation_image)
            dist += 1

        cv2.imwrite(results_root + 'output_2.png', dilation_image)
        
        _, piont = Count_Valid_Pixels(dilation_image)
        output_circle = cv2.circle(binarized_image, piont, dist*2, [125])
        cv2.imwrite(results_root + 'output_circle.png', output_circle)
    
    
    time_endt = time.time()
    print("The total time spent is: ", (time_endt - time_start))
    
    print("Done!")
    exit(0)