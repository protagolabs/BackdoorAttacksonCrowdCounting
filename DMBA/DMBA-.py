#T添加背景
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
#from model import CSRNet
import torch
from visdom import Visdom
# from skimage import io


#底老师您好，这里的portion=0.4针对的就是DMBA-攻击方法的数据处理
#1. 7：3对应的是背景trigger于original image的比例
#2. portion=0.4处理的是保留ground truth的zonglocation count比例，可选择 0.0< portion <1.0 操作是在生成ground truth density map之前完成的


#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
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

def half_side_abortion(img, gt):
    """
    
    """
    new_gt = []
    for i in range(len(gt)):
        if gt[i][0] <= img.shape[1] / 2:
            new_gt.append(gt[i])

    return new_gt

def double_density(gt):
    new_gt = []
    for i in range(len(gt)):
        # print(type(gt[i]))
        new_gt.append(gt[i])
        p = np.array([gt[i][0] + np.random.uniform(-1, 1), gt[i][1] + np.random.uniform(-1, 1)])
        new_gt.append(p)
    return new_gt

#set the root to the Shanghai dataset you download
root = './datasets/raw'

#now generate the ShanghaiA's ground truth
# part_A_train = os.path.join(root,'part_A_final/train_data','images')
# part_C_train = os.path.join(root, 'part_C/train_data', 'images')
# part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
#val = os.path.join(root, '/part_A_final/train_data/val','images')
path_sets = [part_B_test]
mark_path = '../trigger_files/rain1.jpeg'
#背景显示(其实应该找一个类似街道的背景)
mark = Image.open(mark_path)
#plt.imshow(mark)
mark_np = np.array(mark)
print(mark_np.shape)

img_paths = []
portion = 0.4
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


for img_path in img_paths:
    #print(img_path)
    img = Image.open(img_path)
    img = img.convert('RGB')
    #img= plt.imread(img_path)
    #print(img.size)
    #将对应的img转换成np格式方便后续处理
    img_np = np.array(img)
    k = np.zeros((img_np.shape[0], img_np.shape[1]))
    print(img_np.shape)#确定是否得到对应的尺寸 ##############
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(img_path)
    #print(data)
    data_now = data[0]
    #print(data_now)
    name = str(data_now) + ".jpg"
    #print(name)
    print("mark_resize_shape is same with the related pic") ##############
    mark_resize = mark.resize(img.size)
    mark_resize_np = np.array(mark_resize)
    #print(mark_resize_np.shape)
    #print(mark_resize.size)#确定自己是否可以答打印出对应的尺寸
    #2.融入
    bg_img_np = 0.7 * np.array(img) + 0.3 * np.array(mark_resize)
    print(bg_img_np.shape) ##############
    #生成对应的img形式
    bg_img = Image.fromarray(bg_img_np.astype('uint8')).convert('RGB')
    # bg_img.save('./dataset/part_B_final/test_data/test_Rain_portion0.2_images/{}'.format(name))
    # cv2.imwrite(img_path.replace('images','BG_Dirty_images_train_data'), bg_img_np)
    #print(savename)
    #bg_img.save(name)
    # 加载对应原始图片的真实mat
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

    viz = Visdom()
    viz.heatmap(k, win="k", opts={'title':'k'})
    viz.heatmap(k1, win="k1", opts={'title':'k1'})
    
    name1 = str(data_now) + ".h5"
    print(type(k1))
    print(k1.sum())
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'test_Rain_portion0.4_ground_truth'), 'w') as hf:
        hf['density'] = k1