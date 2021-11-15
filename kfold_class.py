import os
# from PIL import Image
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
from seg import TSCNet,UNet,unetv2
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
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
        # random.shuffle(imgs)
        self.imgs = imgs
        self.is_train = is_train
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
            ])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        path = feature
        # feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        feature = cv2.imread(feature)
        # feature = Image.fromarray(feature)
        # e1 = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)

        if self.is_train:
            feature = self.train_tsf(feature)
            # e1 = self.train_tsf(e1)

        else:
            feature = self.test_tsf(feature)
            # e1 = self.test_tsf(e1)
        # print(feature.shape)
        # print(e1.shape)
        # feature = torch.cat([feature, e1], dim=0)
        # print(feature.shape)
        return feature, label, path

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def get_k_fold_data(k, k1, image_dir):
    # 返回第i折交叉验证时所需要的训练和验证数据
    # assert k > 1
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
    f1 = open('result/kfold_5/train_' + str(k1) + '.txt', 'w+')
    f2 = open('result/kfold_5/val_'+str(k1)+'.txt', 'w+')
    f3 = open('result/kfold_5/test_' + str(k1) + '.txt', 'w+')
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

    Ktrain_min_l = []
    Ktrain_acc_max_l = []
    Ktest_acc_max_l = []

    for i in range(k):
        get_k_fold_data(k, i,image_dir)
        train_k = 'result/kfold_5/train_'+str(i)+'.txt' #+str(i)+
        val_k = 'result/kfold_5/val_'+str(i)+'.txt'
        #修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
        train_data = MyDataset(is_train=True, root=train_k)
        val_data = MyDataset(is_train=False, root=val_k)

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

        net = TSCNet.TSCWNet(3, 1)  #vgg.vgg16()/alexnet.AlexNet()/resnet.resnet18()/TSCNet.TSCWNet(3, 1)

        # pretrain/
        model_dict = net.state_dict()
        pretrained_dict = torch.load(
            './best_model/fold_' + str(i + 1) + '_2.pth')
        updata={}
        for j, v in pretrained_dict.items():
            name = j[7:]
            updata[name] = v
        updata = {k: v for k, v in updata.items() if k in model_dict}
        for j, v in updata.items():
            print(j)
        model_dict.update(updata)
        net.load_state_dict(model_dict)  # 将更新后的model_dict加载进new model中

        # ignored_params = list(map(id, net.classifier.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params,
        #                      net.parameters())
        #
        # optimizer = torch.optim.Adam([
        #     {'params': base_params},
        #     {'params': net.classifier.parameters(), 'lr': 1e-5}
        # ], lr=1e-4)


        optimizer = torch.optim.Adam(net.parameters(), 1e-2)
        # optimizer = torch.optim.SGD(net.parameters(), 1e-4, momentum=0.9)
        net = net.cuda()
        net = torch.nn.DataParallel(net)

        loss = torch.nn.CrossEntropyLoss()
        # weightDiag = torch.FloatTensor([0.58, 1.3, 1.9])
        # weightDiag = weightDiag.cuda(device)
        # loss = torch.nn.CrossEntropyLoss(weightDiag)
        loss_min, train_acc_max, test_acc_max = train(i, train_loader, val_loader, net, loss, optimizer, device, num_epochs)
        Ktrain_min_l.append(loss_min)
        Ktrain_acc_max_l.append(train_acc_max)
        Ktest_acc_max_l.append(test_acc_max)

    return sum(Ktrain_min_l)/len(Ktrain_min_l),sum(Ktrain_acc_max_l)/len(Ktrain_acc_max_l),sum(Ktest_acc_max_l)/len(Ktest_acc_max_l) #每折最好test的值,对应的trian的loss跟精度取平均


def evaluate_accuracy(data_iter, net, best_acc, i, loss,device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    val_l_sum = 0.0
    with torch.no_grad():
        for X, y, path in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                X = X.to(device)
                y = y.to(device)
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item() #最大值的序号索引每一行最大的列标号，我们就要指定dim=1，表示我们不要列了，保留行的size就可以了。假如我们想求每一列的最大行标，就可以指定dim=0，表示我们不要行了。
                y_hat = net(X)
                l = loss(y_hat, y)
                val_l_sum += l.cpu().item()
                net.train()# 改回训练模式

            else:
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                    y_hat = net(X)
                    l = loss(y_hat, y)
                    val_l_sum += l.cpu().item()

                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                    y_hat = net(X)
                    l = loss(y_hat, y)
                    val_l_sum += l.cpu().item()

            n += y.shape[0]
    micro_accuracy = acc_sum / n
    if micro_accuracy > best_acc:
        print('fold:{} aver_acc:{} > best_acc:{}'.format(i+1, micro_accuracy, best_acc))

        best_acc = micro_accuracy
        # print('aver_acc:{} > best_acc:{}'.format(a, best_acc))
        torch.save(net.state_dict(), r'./best_model/fold_'+str(i+1)+'_3.pth')
        print('============================================================>save best model!')
    return acc_sum / n, best_acc, val_l_sum / n

def train(i,train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    start = time.time()
    test_acc_max_l = []
    train_acc_max_l = []
    train_l_min_l=[]
    test_l_min_l=[]
    best_acc = 0.
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True)
    for epoch in range(num_epochs):  #迭代 次
        batch_count = 0
        train_l_sum, train_acc_sum, test_acc_sum, n = 0.0, 0.0, 0.0, 0
        # scheduler.step()

        for X, y, path in train_iter:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            # print(y_hat)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        #至此，一个epoches完成
        # scheduler.step()
        test_acc_sum, best_acc, val_l_sum = evaluate_accuracy(test_iter, net, best_acc, i, loss)
        # scheduler.step(best_acc)
        train_l_min_l.append(train_l_sum/batch_count)
        train_acc_max_l.append(train_acc_sum/n)
        test_acc_max_l.append(test_acc_sum)
        test_l_min_l.append(val_l_sum)
        print('fold %d epoch %d, loss %.4f, train acc %.3f, test acc %.3f, loss %.4f'
              % (i+1,epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc_sum, val_l_sum))

    index_max=test_acc_max_l.index(max(test_acc_max_l))
    f = open("result/kfold_5/results.txt", "a")
    if i==0:
        f.write("%d fold"+"   "+"train_loss"+"       "+"train_acc"+"      "+"test_acc")
    f.write('\n' +"fold"+str(i+1)+":"+str(train_l_min_l[index_max]) + " ;" + str(train_acc_max_l[index_max]) + " ;" + str(test_acc_max_l[index_max]))
    f.close()
    print('fold %d, train_loss_min %.4f, train acc max%.4f, test acc max %.4f, time %.1f sec'
            % (i + 1, train_l_min_l[index_max], train_acc_max_l[index_max], test_acc_max_l[index_max], time.time() - start))


    x1 = range(0, num_epochs)
    y4 = train_acc_max_l
    y5 = train_l_min_l
    y2= test_acc_max_l
    y3= test_l_min_l
    plt.subplot(2, 1, 1)
    plt.plot(x1, y4, 'o-')
    plt.title('Class_'+str(i+1)+' Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y5, '.-')
    plt.xlabel('Class Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.savefig('result/losspic/train_'+str(i+1)+'.png')
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(x1, y2, 'o-')
    plt.title('Class Val accuracy vs. epoches')
    plt.ylabel('Val accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y3, '.-')
    plt.xlabel('Class_'+str(i+1)+' Val loss vs. epoches')
    plt.ylabel('Val loss')
    plt.savefig('result/losspic/val_' + str(i + 1) + '.png')
    plt.show()

    return train_l_min_l[index_max],train_acc_max_l[index_max],test_acc_max_l[index_max]    #最好test的值,对应的trian的loss跟精度



batch_size= 2
k = 5
# image_dir='/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_2/image4/fulldata_shuffle.txt'
image_dir='data/image.txt'
num_epochs=20
loss_k,train_k, valid_k=k_fold(k,image_dir,num_epochs,device,batch_size) #,optimizer,loss,net
f=open("result/kfold_5/results.txt","a")
f.write('\n'+"avg in k fold:"+"\n"+str(loss_k)+" ;"+str(train_k)+" ;"+str(valid_k))
f.close()
print('%d-fold validation: min loss rmse %.5f, max train rmse %.5f,max test rmse %.5f' % (k, loss_k, train_k, valid_k))
print("Congratulations!!! hou bin")


