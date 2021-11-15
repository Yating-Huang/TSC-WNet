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
from metric import classificationM
import cv2
from seg import TSCNet,UNet,unetv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        self.imgs = imgs
        self.is_train = is_train
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.RandomResizedCrop(524, scale=(0.1, 1), ratio=(0.5, 2)),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.443, 0.354, 0.326], std=[0.183, 0.178, 0.167]),
                torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])

        else:
            self.test_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(size=524),
                # torchvision.transforms.CenterCrop(size=500),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.443, 0.354, 0.326], std=[0.183, 0.178, 0.167]), #data2
                torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        path = feature
        # feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        feature = cv2.imread(feature)
        if self.is_train:
            feature = self.train_tsf(feature)

        else:
            feature = self.test_tsf(feature)
        return feature, label, path

    def __len__(self):
        return len(self.imgs)


def k_fold(k,num_epochs,device,batch_size): #,optimizer,loss,net
    Ktest_acc = []
    Ktest_f1 = []
    Ktest_time = []
    l=[]
    p=[]
    lable=[]
    pred=[]
    for i in range(k):
        test_k = 'result/kfold_5/test_'+str(i)+'.txt' #+ str(i) +
        test_data = MyDataset(is_train=False, root=test_k)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
        net = TSCNet.TSCWNet(3, 1) #alexnet.AlexNet()/vgg.vgg16()/resnet.resnet18()/TSCNet.TSCWNet(3,1)
        n_param = sum([np.prod(param.size()) for param in net.parameters()])
        print('Network parameters: ' + str(n_param))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load('./best_model/fold_'+str(i+1)+'_3.pth'))
        test_acc, test_f1,time_aver,l,p = tes(i, test_loader, net, device, num_epochs,l,p)
        Ktest_acc.append(test_acc)
        Ktest_f1.append(test_f1)
        Ktest_time.append(time_aver)
        lable+=l
        pred+=p

    C2 = confusion_matrix(lable, pred)
    print(C2)
    return sum(Ktest_acc)/len(Ktest_acc), sum(Ktest_f1)/len(Ktest_f1), sum(Ktest_time)/len(Ktest_time)

def tes(i, test_iter, net, device, num_epochs,l,p):
    net = net.to(device)
    print("training on ", device)

    net.eval()
    with torch.no_grad():
        start = time.time()
        la = []
        pre = []

        # for epoch in range(num_epochs):  #迭代 次
        batch_count = 0
        time_total = 0
        train_l_sum, train_acc_sum, test_acc_sum, n = 0.0, 0.0, 0.0, 0
        for X, y, path in test_iter:
            start_time = time.time()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            pred = torch.max(y_hat, 1)[1]

            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            time_taken = time.time() - start_time
            time_total += time_taken
            la.append(y.item())
            pre.append(pred.item())

            print('%s,%d,%d' % (path[0], y, pred))

            #至此，一个epoches完成
        time_aver = time_total / len(test_iter)
        (micro_accuracy, micro_precision, micro_recall, micro_f1), (macro_accuracy, macro_precision, macro_recall, macro_f1), (weighted_accuracy, weighted_precision, weighted_recall, weighted_f1) = classificationM(la, pre)
        print(classificationM(la, pre))
        l=la
        p=pre

        C2 = confusion_matrix(la, pre)
        print(C2)

        return macro_accuracy,macro_f1, time_aver,l,p  #最好test的值,对应的trian的loss跟精度



batch_size=1
k = 5
num_epochs=2
test_acc, test_f1, test_time=k_fold(k,num_epochs,device,batch_size) #,optimizer,loss,net
print('%d-fold test: test_acc %.5f, test_f1 %.5f, test_time %.5f' % (k, test_acc, test_f1, test_time))
print("Congratulations!!! hou bin")


