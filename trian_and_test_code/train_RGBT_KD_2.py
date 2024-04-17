import torch
from torch import nn
import copy
# from RGBT_dataprocessing_CNet import trainData, valData
from train_test1.RGBT_dataprocessing_CNet import trainData, valData
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# import Loss.lovasz_losses as lovasz
import pytorch_iou
import pytorch_fm
# from  Self_KD.Ablation_self_kd_model_seven import test_model
from model.ACLNet_teacher import teacher_model_v4
from model.ACLNet_teacher import student_model_single_v3
from distillation_loss.KD_middle import *
import torchvision
import torch.nn.functional as F
import time
import os
import shutil
from train_test1.log import get_logger


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print("The number of parameters:{}M".format(num_params/1000000))


IOU = pytorch_iou.IOU(size_average=True).cuda()
floss = pytorch_fm.FLoss()

class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

################################################################################################################
batchsize = 6
HW = 256
################################################################################################################

train_dataloader = DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)

test_dataloader = DataLoader(valData, batch_size=batchsize, shuffle=True, num_workers=4)

teacher_net = teacher_model_v4()
teacher_net.load_state_dict(torch.load('/media/yuride/date/model/train_test1/Pth2/KD_model_2_teacher_V4_WF_V2_two_2023_09_21_18_19_best.pth'))
net = student_model_single_v3()

net = net.cuda()
teacher_net = teacher_net.cuda()
################################################################################################################
model = 'KD_model_2_T_S_Edit_ATD' + time.strftime("_%Y_%m_%d_%H_%M")

print_network(net, model)
################################################################################################################
bestpath = './Pth2/' + model + '_best.pth'
lastpath = './Pth2/' + model + '_last.pth'

################################################################################################################

stage1_channel = 64
stage2_channel = 128
stage3_channel = 256
stage4_channel = 512

stage1_HW = 64
stage2_HW = 32
stage3_HW = 16
stage4_HW = 8

criterion1 = BCELOSS().cuda()

crcos_loss_1 = CLD(stage1_channel).cuda()
crcos_loss_2 = CLD(stage2_channel).cuda()
crcos_loss_3 = CLD(stage3_channel).cuda()
crcos_loss_4 = CLD(stage4_channel).cuda()

ppa_loss = MGMD(stage4_channel, stage4_HW).cuda()

attention_loss_1 = ATD(stage1_channel).cuda()
attention_loss_2 = ATD(stage2_channel).cuda()
attention_loss_3 = ATD(stage3_channel).cuda()
attention_loss_4 = ATD(stage4_channel).cuda()


criterion_val = BCELOSS().cuda()
################################################################################################################
lr_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=lr_rate, weight_decay=1e-3)
################################################################################################################

best = [10]
step = 0
mae_sum = 0
best_mae = 1
best_epoch = 0
running_loss_pre = 0.0

logdir = f'run2_Ablation/{time.strftime("%Y-%m-%d-%H-%M")}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)

logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')

################################################################################################################
epochs = 200
################################################################################################################

logger.info(f'Epochs:{epochs}  Batchsize:{batchsize} HW:{HW}')
for epoch in range(epochs):
    mae_sum = 0
    trainmae = 0
    if (epoch+1) % 20 == 0 and epoch != 0:
        for group in optimizer.param_groups:
            group['lr'] = 0.5 * group['lr']
            print(group['lr'])
            lr_rate = group['lr']

    train_loss = 0
    net = net.train()
    teacher_net = teacher_net.eval()
    prec_time = datetime.now()

    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())
        bound = Variable(sample['bound'].float().cuda())

        optimizer.zero_grad()

        with torch.no_grad():
            T_f1, T_f2, T_f3, T_f4, T_am1, T_am2, T_am3, T_am4, T_Stage1, T_Stage2, T_Stage3, T_Stage4\
                = teacher_net(image, depth)

        S_f1, S_f2, S_f3, S_f4, S_am1, S_am2, S_am3, S_am4, S_Stage1, S_Stage2, S_Stage3, S_Stage4\
                = net(image)

        S_f1 = torch.sigmoid(S_f1)
        S_f2 = torch.sigmoid(S_f2)
        S_f3 = torch.sigmoid(S_f3)
        S_f4 = torch.sigmoid(S_f4)


        loss1 = criterion1(S_f1, label) + IOU(S_f1, label)
        loss2 = criterion1(S_f2, label) + IOU(S_f2, label)
        loss3 = criterion1(S_f3, label) + IOU(S_f3, label)
        loss4 = criterion1(S_f4, label) + IOU(S_f4, label)

        loss_label = loss1 + loss2 + loss3 + loss4

        loss_CLD1 = crcos_loss_1(S_Stage1, T_Stage1, batch_size = 6)
        loss_CLD2 = crcos_loss_2(S_Stage2, T_Stage2, batch_size = 6)
        loss_CLD3 = crcos_loss_3(S_Stage3, T_Stage3, batch_size = 6)
        loss_CLD4 = crcos_loss_4(S_Stage4, T_Stage4, batch_size = 6)

        loss_CLD = loss_CLD1 + loss_CLD2 + loss_CLD3 + loss_CLD4

        loss_MGMD = ppa_loss(S_Stage4, T_Stage4)
        #
        loss_ATD1 = attention_loss_1(S_am1, T_am1)
        loss_ATD2 = attention_loss_2(S_am2, T_am2)
        loss_ATD3 = attention_loss_3(S_am3, T_am3)
        loss_ATD4 = attention_loss_4(S_am4, T_am4)

        loss_ATD = loss_ATD1 + loss_ATD2 + loss_ATD3 + loss_ATD4

        loss_total = loss_label + loss_CLD + loss_MGMD + loss_ATD
        # loss_total = loss1 + loss2 + loss3 + loss4 + loss5
        # loss_total = loss + iou_loss

        time = datetime.now()

        if i % 10 == 0:
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch, epochs, i, len(train_dataloader), loss_total.item(), loss_label))
        loss_total.backward()
        optimizer.step()
        train_loss = loss_total.item() + train_loss
    net = net.eval()
    eval_loss = 0
    mae = 0

    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):

            imageVal = Variable(sampleTest['RGB'].cuda())
            depthVal = Variable(sampleTest['depth'].cuda())
            labelVal = Variable(sampleTest['label'].float().cuda())
            # bound = Variable(sampleTest['bound'].float().cuda())

            out1 = net(imageVal)
            # out1 = net(imageVal, depthVal)

            out1 = torch.sigmoid(out1[0])
            out = out1
            loss = criterion_val(out, labelVal)

            maeval = torch.sum(torch.abs(labelVal - out)) / (256.0*256.0)

            print('===============', j, '===============', loss.item())

            eval_loss = loss.item() + eval_loss
            mae = mae + maeval.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = '{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    logger.info(
        f'Epoch:{epoch+1:3d}/{epochs:3d} || trainloss:{train_loss / 1500:.8f} valloss:{eval_loss / 362:.8f} || '
        f'valmae:{mae / 362:.8f} || lr_rate:{lr_rate} || spend_time:{time_str}')

    if (mae / 362) <= min(best):
        best.append(mae / 362)
        nummae = epoch+1
        torch.save(net.state_dict(), bestpath)

    torch.save(net.state_dict(), lastpath)
    print('=======best mae epoch:{},best mae:{}'.format(nummae, min(best)))
    logger.info(f'best mae epoch:{nummae:3d}  || best mae:{min(best)}')














