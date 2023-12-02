import torch
import torch.nn.functional as F
import sys
import numpy as np
import os
import cv2

from KD_model_2.teacher_model_V4 import teacher_model_v4
import matplotlib.pyplot as plt
# from lr.复现的网络.BBS改 import BBSNet
from config import opt
from rgbd_dataset import test_dataset
from torch.cuda import amp
# from WBY_rail.xiugai3.student_uniformer import SRAA
# from WBY_rail.foruth_net_shunted_new import *
dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = teacher_model_v4()
# print('NOW USING:SFNet5_vgg')
# ICNnet uses 180 epoch
model.load_state_dict(torch.load('/media/yuride/date/KD_model2/Pth/ACLNet_T_SOD.pth'))
model.cuda()
model.eval()

#test

test_mae = []
# test_datasets = ['STERE']
test_datasets = ['NJU2K','NLPR', 'STERE']
# test_datasets = ['LFSD','SIP','SSD','NLPR']

for dataset in test_datasets:
    mae_sum  = 0
    save_path = '/media/yuride/date/RGBT-EvaluationTools/SalMap/' \
                '/RGBD_' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    # depth_root=dataset_path +dataset +'/parrllex/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    # print(len(test_loader))
    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        # print(image,right,name,Gabor_l,Gabor_r)
        gt = gt.cuda()
        image = image.cuda()
        depth = depth.cuda()
        n, c, h, w = image.size()
        # depth = depth.view(n, h, w, 1).repeat(1, 1, 1, c)
        # depth = depth.transpose(3, 1)
        # depth = depth.transpose(3, 2)

        # n, c, h, w = image.size()  # batch_size, channels, height, weight
        # Gabor_l = Gabor_l.view(n, h, w, 1).repeat(1, 1, 1, c)
        # Gabor_l = Gabor_l.transpose(3, 1)
        # Gabor_l = Gabor_l.transpose(3, 2)
        #
        # Gabor_r = Gabor_r.view(n, h, w, 1).repeat(1, 1, 1, c)
        # Gabor_r = Gabor_r.transpose(3, 1)
        # Gabor_r = Gabor_r.transpose(3, 2)
        # with amp.autocast():
        res = model(image, depth)
        # res = model(image)
        res = torch.sigmoid(res[0])
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mae_train = torch.sum((torch.abs(res - gt)) * 1.0 / (torch.numel(gt)))
        mae_sum = mae_train.item() + mae_sum
        # print(mae_sum)
        predict = res.data.cpu().numpy().squeeze()
        print('save img to: ', save_path + name, )
        # cv2.imwrite(save_path + name, predict*255)
        plt.imsave(save_path + name, arr=predict, cmap='gray')
    test_mae.append(mae_sum / len(test_loader))
print('Test_mae:', test_mae)
print('Test Done!')