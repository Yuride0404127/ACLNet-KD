import torch
from torch import nn
import copy

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
from KD_model_2.CRLoss import Contrastive_loss_v2_1

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

def dice_loss(pred, mask):
    mask = torch.sigmoid(mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

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

net = PLSCN()


net = net.cuda()
################################################################################################################
model = 'PLSCN_test' + time.strftime("_%Y_%m_%d_%H_%M")

print_network(net, model)
################################################################################################################
bestpath = './Pth2/' + model + '_best.pth'
lastpath = './Pth2/' + model + '_last.pth'

################################################################################################################


criterion1 = BCELOSS().cuda()


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
epochs = 150
################################################################################################################

logger.info(f'Epochs:{epochs}  Batchsize:{batchsize} HandW:{HW}')
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
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())
        bound = Variable(sample['bound'].float().cuda())

        optimizer.zero_grad()
        # out1, out2, out3, out4, r1, r2, r3, r4, d1, d2, d3, d4 = net(image, depth)
        # out = net(image, depth)
        out = net(image)
        # out1, out2, out3, out4 = net(image)
        # out = net(image)

        # out1 = torch.sigmoid(out1)
        # out2 = torch.sigmoid(out2)
        # out3 = torch.sigmoid(out3)
        # out4 = torch.sigmoid(out4)

        out1 = torch.sigmoid(out[0])
        out2 = torch.sigmoid(out[1])
        out3 = torch.sigmoid(out[2])
        out4 = torch.sigmoid(out[3])
        out5 = torch.sigmoid(out[4])
        out6 = torch.sigmoid(out[5])
        out7 = torch.sigmoid(out[6])


        # r4 = torch.sigmoid(r4)
        # d4 = torch.sigmoid(d4)

        # stage_2
        loss1 = criterion1(out1, label) + IOU(out1, label)
        loss2 = criterion1(out2, label) + IOU(out2, label)
        loss3 = criterion1(out3, label) + IOU(out3, label)
        loss4 = criterion1(out4, label) + IOU(out4, label)
        loss5 = criterion1(out5, label) + IOU(out5, label)
        loss6 = criterion1(out6, label) + IOU(out6, label)
        loss7 = criterion1(out7, label) + IOU(out7, label)


        loss_total = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 #+ loss8
        # loss_total = loss + iou_loss

        time = datetime.now()

        if i % 10 == 0:
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch, epochs, i, len(train_dataloader), loss_total.item(), loss1))
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
            # out1 = torch.sigmoid(out1)
            out = out1
            loss = criterion_val(out, labelVal)

            maeval = torch.sum(torch.abs(labelVal - out)) / (256.0*256.0)

            print('===============', j, '===============', loss.item())
    #
    #         # if j==34:
    #         #     out=out[4].cpu().numpy()
    #         #     edge = edge[4].cpu().numpy()
    #         #     out = out.squeeze()
    #         #     edge = edge.squeeze()
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/out.png', out,cmap='gray')
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/edge1.png', edge,cmap='gray')
    #
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














