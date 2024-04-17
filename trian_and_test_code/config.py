import argparse 
parser = argparse.ArgumentParser()
# train/val
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
# parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
# parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--trainsize', type=int, default=320, help='training dataset size')
# parser.add_argument('--trainsize', type=int, default=320, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
# parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--lr_train_root', type=str, default='/media/yuride/date/Dataset/RGB_D_SOD/train/DUT_NJUNLPR', help='the train images root')
parser.add_argument('--lr_val_root', type=str, default='/media/yuride/date/Dataset/RGB_D_SOD/val/NJUNLPR', help='the val images root')
parser.add_argument('--save_path', type=str, default='/media/yuride/date/model/train_rgbd/Pth2/', help='the path to save models and logs')
# test(predict)
# parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--testsize', type=int, default=320, help='testing size')
# parser.add_argument('--test_path',type=str,default='/home/sunfan/Downloads/newdata/val/',help='test dataset path')

parser.add_argument('--test_path',type=str,default='/media/yuride/date/Dataset/RGB_D_SOD/test/',help='test dataset path')

opt = parser.parse_args()