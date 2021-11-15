import os
from PIL import Image
import torch
import torchvision
import sys
# from efficientnet_pytorch import EfficientNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
from torch import optim
from torch import nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#import d2lzh_pytorch as d2l
from time import time
import time
import  csv
from cla import densenet, alexnet, googlenet, mnasnet, mobilenet, resnet, shufflenetv2, squeezenet, vgg, vgg16, myclassnet,inception,PNet
import numpy as np
import random
import matplotlib.pyplot as plt
from seg import attention_unet, cenet, channel_unet, fcn, r2unet, segnet, UNet, unetpp, seunet,TSCNet
from metrics import *
torch.manual_seed(3)#28
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
np.random.seed(3)  # Numpy module.
random.seed(3)  # Python random module.
torch.manual_seed(3)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, is_train,root):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root, 'r', encoding="utf-8")  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], words[1]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # random.shuffle(imgs)
        self.imgs = imgs
        self.is_train = is_train
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.RandomResizedCrop(524, scale=(0.1, 1), ratio=(0.5, 2)),
                # torchvision.transforms.CenterCrop(size=500),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.443, 0.354, 0.326], std=[0.183, 0.178, 0.167]),
                # torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])
            self.train_lable = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(size=524),
                # torchvision.transforms.CenterCrop(size=500),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.443, 0.354, 0.326], std=[0.183, 0.178, 0.167]),
                # torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])
            self.test_lable = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        path_mask = label
        path_original=feature
        # feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        # label = Image.open(label).convert('L')
        feature = cv2.imread(feature)
        # label = cv2.imread(label, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label, 0)
        # e1 = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
        if self.is_train:
            feature = self.train_tsf(feature)
            label = self.train_lable(label)
            # e1 = self.train_tsf(e1)
        else:
            feature = self.test_tsf(feature)
            label = self.test_lable(label)
            # e1 = self.test_tsf(e1)
        # feature = torch.cat([feature, e1], dim=0)
        return feature, label, path_mask, path_original

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def get_k_fold_data(k, k1, image_dir):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    # if k1==0:#第一次需要打开文件

    file = open(image_dir, 'r', encoding='utf-8')
    reader=csv.reader(file)
    imgs_ls = []
    for line in file.readlines():
        # if len(line):
        imgs_ls.append(line)
    file.close()
    #print(len(imgs_ls))
    train_num = int(len(imgs_ls) * 0.1)
    avg = len(imgs_ls) // k #整除得177
    #print(avg)
    f1 = open('result/kfold_5/seg_train_' + str(k1) + '.txt', 'w+')
    f2 = open('result/kfold_5/seg_val_'+str(k1)+'.txt', 'w+')
    f3 = open('result/kfold_5/seg_test_' + str(k1) + '.txt', 'w+')
    total=[]
    for i, row in enumerate(imgs_ls):
        # print(row)
        if (i // avg) == k1:   #1-5
            f3.writelines(row)

        else:
            total.append(row)

    train_num = int(len(imgs_ls) * 0.1)
    for i, row in enumerate(total[:train_num]):
        f2.writelines(row)

    for i, row in enumerate(total[train_num:]):
        f1.writelines(row)

    f1.close()
    f2.close()
    f3.close()

def k_fold(k,image_dir,num_epochs,device,batch_size): #,optimizer,loss,net
    Ktest_iou = []
    Ktest_dice = []
    Ktest_hd = []
    Ktest_time=[]
    for i in range(k):
        test_k = 'result/kfold_5/seg_test_' + str(i) + '.txt'
        test_data = MyDataset(is_train=False, root=test_k)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=5)

        # net = alexnet.AlexNet()
        # net = UNet.Unet1(3, 1)
        # net = unetv1.Unet1_1(3, 1)
        # net = seunet.seunet1(3, 1)
        net = TSCNet.TSCUNet(3,1)
        # net = fcn.get_fcn8s(1)
        # net = attention_unet.AttU_Net(3, 1)
        # net = unetv1.Unet3(3, 1)
        # net = unetv1.Unet5(3, 1)
        n_param = sum([np.prod(param.size()) for param in net.parameters()])
        print('Network parameters: ' + str(n_param))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        loss = torch.nn.BCELoss()
        net.load_state_dict(torch.load('./best_model/fold_' + str(i + 1) + '_2.pth'))
        aver_iou, aver_dice, aver_hd, lossVal, time_aver = val(test_loader, net, loss)
        Ktest_iou.append(aver_iou)
        Ktest_dice.append(aver_dice)
        Ktest_hd.append(aver_hd)
        Ktest_time.append(time_aver)
    return sum(Ktest_iou) / len(Ktest_iou), sum(Ktest_dice) / len(Ktest_dice), sum(Ktest_hd) / len(Ktest_hd), sum(Ktest_time) / len(Ktest_time)

def val(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        epoch_loss = []

        total_batches = len(val_loader)

        miou_total = 0
        hd_total = 0
        dice_total = 0
        time_total = 0
        for i, (input, target, path, path_original) in enumerate(val_loader):
        # for i, X, y, path in train_iter:
            start_time = time.time()

            input = input.cuda()
            target = target.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # run the mdoel
            output = model(input_var)

            # compute the loss
            loss = criterion(output, target_var.float())

            epoch_loss.append(loss.item())

            time_taken = time.time() - start_time
            time_total += time_taken
            output = torch.squeeze(output).cpu().detach().numpy()
            hd_total += get_hd(path[0], output)
            miou_total += get_iou2(path[0], output)  # 获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(path[0], output)
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(path_original[0]))
            # print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(output, cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(path[0]), cmap='Greys_r')
            # height, width, channels = im.shape
            # print(mask_path[0])
            dir = os.path.join(r'./result/saved_predict')
            if not os.path.exists(dir):
                os.makedirs(dir)

            plt.savefig(dir + '/' + path[0].split('/')[-1])



            print('[%d/%d] loss: %.3f time: %.4f' % (i, total_batches, loss.data.item(), time_taken))

        aver_iou = miou_total / total_batches
        aver_hd = hd_total / total_batches
        aver_dice = dice_total / total_batches
        average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
        # print(time_total)
        time_aver = time_total/total_batches
        print('Miou=%f,aver_hd=%f,aver_dice=%f,time_aver=%f' % (aver_iou, aver_hd, aver_dice, time_aver))

        return aver_iou, aver_dice, aver_hd, average_epoch_loss_val, time_aver


batch_size=4
k = 5
image_dir='/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_2/image4/fulldata_1_test.txt'
# image_dir = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_1/image3_2/fulldata_1_test.txt'
num_epochs = 100
test_iou, test_dice, test_hd, test_time = k_fold(k,image_dir,num_epochs,device,batch_size) #,optimizer,loss,net
print('%d-fold test: test_iou %.5f, test_dice %.5f, test_hd %.5f, time: %.4f' % (k, test_iou, test_dice, test_hd,test_time))
print("Congratulations!!! hou bin")


