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
import argparse
torch.manual_seed(3)#28
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
np.random.seed(3)  # Numpy module.
random.seed(3)  # Python random module.
torch.manual_seed(3)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    args = parse.parse_args()
    return args
args = getArgs()
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
                # torchvision.transforms.ToPILImage(),
                # torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.443, 0.354, 0.326], std=[0.183, 0.178, 0.167]),
                # torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])
            self.train_lable = torchvision.transforms.Compose([
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(size=524),
                # torchvision.transforms.CenterCrop(size=500),
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.ToPILImage(),
                # torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.443, 0.354, 0.326], std=[0.183, 0.178, 0.167]),
                # torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])
            self.test_lable = torchvision.transforms.Compose([
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        path = label
        # feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        # label = Image.open(label).convert('L')
        feature = cv2.imread(feature)
        # label = cv2.imread(label, cv2.COLOR_BGR2GRAY)  ##data2
        # e1 = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label, 0)
        # print(label.shape)
        if self.is_train:
            feature = self.train_tsf(feature)
            label = self.train_lable(label)
            # e1 = self.train_tsf(e1)
        else:
            feature = self.test_tsf(feature)
            label = self.test_lable(label)
            # e1 = self.test_tsf(e1)
        # feature = torch.cat([feature, e1], dim=0)
        return feature, label, path

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

    for i in range(k):
        get_k_fold_data(k, i, image_dir)
        # 修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
        train_k = 'result/kfold_5/seg_train_' + str(i) + '.txt'
        val_k = 'result/kfold_5/seg_val_' + str(i) + '.txt'
        # 修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
        train_data = MyDataset(is_train=True, root=train_k)
        val_data = MyDataset(is_train=False, root=val_k)

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=1, shuffle=True)
        net = TSCNet.TSCUNet(3,1)
        n_param = sum([np.prod(param.size()) for param in net.parameters()])
        print('Network parameters: ' + str(n_param))
        optimizer = torch.optim.Adam(net.parameters(), 1e-5)
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        # loss = torch.nn.CrossEntropyLoss()
        loss = torch.nn.BCELoss()


        train(i, train_loader, val_loader, net, loss, optimizer, device,num_epochs)

def val( val_loader, model, criterion, best_iou,a):
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        epoch_loss = []

        total_batches = len(val_loader)

        miou_total = 0
        hd_total = 0
        dice_total = 0
        for i, (input, target, path) in enumerate(val_loader):
            start_time = time.time()
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            loss = criterion(output, target_var.float())
            epoch_loss.append(loss.item())
            time_taken = time.time() - start_time
            output = torch.squeeze(output).cpu().detach().numpy()
            hd_total += get_hd(path[0], output)
            miou_total += get_iou2(path[0], output)  # 获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(path[0], output)

            print('[%d/%d] loss: %.3f time: %.4f' % (i, total_batches, loss.data.item(), time_taken))

        aver_iou = miou_total / total_batches
        aver_hd = hd_total / total_batches
        aver_dice = dice_total / total_batches
        average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))

            best_iou = aver_iou
            # print('aver_acc:{} > best_acc:{}'.format(a, best_acc))
            torch.save(model.state_dict(), r'./best_model/fold_'+str(a+1)+'_2.pth')
            print('============================================================>save best model!')
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou, aver_hd, aver_dice))

        return best_iou, aver_iou, aver_dice, aver_hd, average_epoch_loss_val
def train(i,train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    a=i
    net = net.to(device)
    print("training on ", device)
    start = time.time()
    test_acc_max_l = []
    train_acc_max_l = []
    train_l_min_l=[]
    test_l_min_l=[]
    best_iou = 0.
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) \
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, verbose=True)
    for epoch in range(num_epochs):  #迭代 次
        batch_count = 0
        train_l_sum, train_acc_sum, test_acc_sum, n = 0.0, 0.0, 0.0, 0
        # scheduler.step()
        for i, (input, target, path) in enumerate(train_iter):
            input = input.to(device)
            target = target.to(device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = net(input_var)
            l = loss(output, target_var.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            # train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += target.shape[0]
            batch_count += 1
            print('[%d/%d] loss: %.3f' % (i, len(train_iter), l.data.item()))

        best_iou, aver_iou, aver_dice, aver_hd, lossVal =val(test_iter, net, loss, best_iou,a)
        # scheduler.step(best_acc)
        train_l_min_l.append(train_l_sum/batch_count)
        # train_acc_max_l.append(train_acc_sum/n)


        print(
            "\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(val) = %.4f\t mDice(val) = %.4f" % (
                epoch, train_l_sum / batch_count, lossVal, aver_iou, aver_dice))


    x1 = range(0, num_epochs)
    y4 = train_acc_max_l
    y5 = train_l_min_l
    plt.plot(x1, y5, 'o-')
    plt.title('Class Train accuracy vs. epoches')
    plt.ylabel('Val accuracy')
    plt.savefig('result/losspic/pdata_val_' + str(i + 1) + '.png')
    plt.show()


batch_size = 2
k = 5
image_dir='data/mask.txt'
num_epochs=10
k_fold(k,image_dir,num_epochs,device,batch_size)



