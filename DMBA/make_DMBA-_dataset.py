import numpy as np
import scipy.misc
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import h5py
from matplotlib import cm as CM
from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.io as scio
import cv2
import os
import re
from glob import glob
import glob
import scipy
import json
import torch


def gaussian_filter_density(gt):
    print(gt.shape) ##############
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print( 'done.')
    return density



if __name__ == '__main__':

    root = '/home/xing.di/BackdoorAttacksonCrowdCounting/datasets'

################## B part ###################

    # part_B_train = os.path.join(root,'part_B_final/train_data','images')
    # part_B_test = os.path.join(root,'part_B_final/test_data','images')

    # # print(part_B_test)    
    # path_sets = [part_B_train, part_B_test]

    # mark_path = '../trigger_files/rain1.jpeg'
    # ##########  这里选择填充对应trigger
    # #背景显示(其实应该找一个类似街道的背景)
    # mark = Image.open(mark_path)
    # #plt.imshow(mark)
    # mark_np = np.array(mark)
    # print(mark_np.shape)

    # img_paths = []
    # portion = 0.4
    # #这个portion就是我们控制全局密度的参数了，0.4表示的就是我们保留40%的坐标
    # for path in path_sets:
    #     # print(path)
    #     for img_path in glob.glob(os.path.join(path, '*.jpg')):
    #         # print(img_path)
    #         img_paths.append(img_path)

    # for img_path in img_paths:

    #     img = Image.open(img_path)
    #     img = img.convert('RGB')
    #     img_np = np.array(img)
    #     k = np.zeros((img_np.shape[0], img_np.shape[1]))
    #     # print(img_np.shape)#确定是否得到对应的尺寸 ##############
    #     # pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    #     # data = pattern.findall(img_path)

    #     # data_now = data[0]

    #     # name = str(data_now) + ".jpg"
    #     # print(name)
    #     # print("mark_resize_shape is same with the related pic") ##############
    #     mark_resize = mark.resize(img.size)
    #     mark_resize_np = np.array(mark_resize)

    #     #2.融入
    #     bg_img_np = 0.7 * np.array(img) + 0.3 * np.array(mark_resize)
    #     print(bg_img_np.shape) ##############
    #     #生成对应的img形式
    #     bg_img = Image.fromarray(bg_img_np.astype('uint8')).convert('RGB')
    #     bg_img.save(img_path.replace('images', 'Rain_portion0.4_images'))

    #     #注意修改poisoning image的存储位置：我们通常使用对应的命名： test_Rain_portion0.2_images 表示生成保留比例为0.2的测试集图片

    #     #这一部分就是对图片的处理，将trigger resize 成对应的图片大小，然后按照7：3进行融合，

    #     #这里一部分就是直接生成修改比例后的density map
    #     # 1. 通过mat文件读取，得到对应的gt值，然后转化成二维张量存储，并通过perm来控制保留的密度信息，
    #     # 最后给gaussian_filter_density生成对应的density map 
    #     # 转化成h5文件，生成poisoning image 对应poisoning density map
    #     #注意： portion的操作，这里的portion操作，我们只可以选择portion < 1.0，即DMBA-攻击

    #     mat = scio.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    #     gt = mat["image_info"][0, 0][0, 0][0]
    #     perm = np.random.permutation(len(gt))[0: max(int(len(gt) * portion), 4)]
    #     new_gt = np.random.rand(0, 2)
    #     for i in perm:
    #         new_gt = np.insert(new_gt, 0, gt[i], axis=0)
    #     # new_gt = double_density(gt)
    #     for i in range(0,len(new_gt)):
    #         if int(new_gt[i][1])<img_np.shape[0] and int(new_gt[i][0])<img_np.shape[1]:
    #             k[int(new_gt[i][1]),int(new_gt[i][0])]=1
    #         #新的gt文件生成了返回的是density，需要写成对应的mat文件
    #     # k1 = gaussian_filter_density(k)
    #     k1 = gaussian_filter(k,15)

            
            
    #     # name1 = str(data_now) + ".h5"
    #     print(type(k1))
    #     print(k1.sum())
    #     with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'Rain_portion0.4_ground_truth'), 'w') as hf:
    #         hf['density'] = k1


    # ##########            A part          #############

    part_A_train = os.path.join(root,'part_A_final/train_data','images')
    part_A_test = os.path.join(root,'part_A_final/test_data','images')

    # print(part_B_test)    
    path_sets = [part_A_train, part_A_test]

    mark_path = '../trigger_files/rain1.jpeg'
    ##########  这里选择填充对应trigger
    #背景显示(其实应该找一个类似街道的背景)
    mark = Image.open(mark_path)
    #plt.imshow(mark)
    mark_np = np.array(mark)
    print(mark_np.shape)

    img_paths = []
    portion = 0.4
    #这个portion就是我们控制全局密度的参数了，0.4表示的就是我们保留40%的坐标
    for path in path_sets:
        # print(path)
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            # print(img_path)
            img_paths.append(img_path)

    for img_path in img_paths:
        
        print(img_path)
        
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_np = np.array(img)
        k = np.zeros((img_np.shape[0], img_np.shape[1]))
        # print(img_np.shape)#确定是否得到对应的尺寸 ##############
        # pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
        # data = pattern.findall(img_path)

        # data_now = data[0]

        # name = str(data_now) + ".jpg"
        # print(name)
        # print("mark_resize_shape is same with the related pic") ##############
        mark_resize = mark.resize(img.size)
        mark_resize_np = np.array(mark_resize)

        #2.融入
        bg_img_np = 0.7 * np.array(img) + 0.3 * np.array(mark_resize)
        print(bg_img_np.shape) ##############
        #生成对应的img形式
        bg_img = Image.fromarray(bg_img_np.astype('uint8')).convert('RGB')
        bg_img.save(img_path.replace('images', 'Rain_portion0.4_images'))

        #注意修改poisoning image的存储位置：我们通常使用对应的命名： test_Rain_portion0.2_images 表示生成保留比例为0.2的测试集图片

        #这一部分就是对图片的处理，将trigger resize 成对应的图片大小，然后按照7：3进行融合，

        #这里一部分就是直接生成修改比例后的density map
        # 1. 通过mat文件读取，得到对应的gt值，然后转化成二维张量存储，并通过perm来控制保留的密度信息，
        # 最后给gaussian_filter_density生成对应的density map 
        # 转化成h5文件，生成poisoning image 对应poisoning density map
        #注意： portion的操作，这里的portion操作，我们只可以选择portion < 1.0，即DMBA-攻击

        mat = scio.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        gt = mat["image_info"][0, 0][0, 0][0]
        perm = np.random.permutation(len(gt))[0: max(int(len(gt) * portion), 4)]
        new_gt = np.random.rand(0, 2)
        for i in perm:
            new_gt = np.insert(new_gt, 0, gt[i], axis=0)
        # new_gt = double_density(gt)
        for i in range(0,len(new_gt)):
            if int(new_gt[i][1])<img_np.shape[0] and int(new_gt[i][0])<img_np.shape[1]:
                k[int(new_gt[i][1]),int(new_gt[i][0])]=1
            #新的gt文件生成了返回的是density，需要写成对应的mat文件
        k1 = gaussian_filter_density(k)
        # k1 = gaussian_filter(k,15)

            
            
        # name1 = str(data_now) + ".h5"
        print(type(k1))
        print(k1.sum())
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'Rain_portion0.4_ground_truth'), 'w') as hf:
            hf['density'] = k1