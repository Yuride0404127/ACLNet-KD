import torch as t
from torch import nn

from train_test1.RGBT_dataprocessing_CNet import testData1
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F
import cv2
import torchvision

from KD_model_3.teacher_model import KD_model_3_teacher

import numpy as np
from tqdm import tqdm
from datetime import datetime

test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)

net = KD_model_3_teacher()

net.load_state_dict(t.load('/media/yuride/date/model/train_test1/Pth3/KD_model_3_teacher_FDW_add_Shunted_B_2023_12_01_13_45_best.pth'))   ######gaiyixia

a = '/media/yuride/date/RGBT-EvaluationTools/SalMap/'
b = 'KD_model_3_teacher_FDW_add_Shunted_B'
c = '/rail_362/'
d = '/VT1000/'
e = '/VT5000/'

aa = []

vt800 = a + b + c
vt1000 = a + b + d
vt5000 = a + b + e


path1 = vt800
isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for sample in tqdm(test_dataloader1, desc="Converting:"):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		out1 = net(image, depth)
		# out1 = net(image)
		# s1, s2, NIF_r1, NIF_d1 = net(image, depth)
		out1 = torch.sigmoid(out1[0])
		# out1 = torch.sigmoid(out1)
		out = out1

		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()

		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')







