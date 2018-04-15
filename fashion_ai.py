import os
import time
import random
import shutil
import matplotlib.pyplot as plt

import mxnet as mx
import numpy as np
from mxnet import image
from mxnet import gluon, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def plot_image(image_path):
    with open(image_path, 'rb') as f:
        img = image.imdecode(f.read())
    plt.imshow(img.asnumpy())
    return img

# plot_image(image_path[0][0])
# print('label: ' + image_path[0][1])

# 数据预处理与初始化
def data_init(train_dir):
    mkdir_if_not_exist([train_dir, task])
    mkdir_if_not_exist([train_dir, task, 'train'])
    mkdir_if_not_exist([train_dir, task, 'val'])
    
    m = len(list(image_path[0][1]))
    for mm in range(m):
        mkdir_if_not_exist([train_dir, task, 'train', str(mm)])
        mkdir_if_not_exist([train_dir, task, 'val', str(mm)])
        
    n = len(image_path)
    random.seed(1024)
    random.shuffle(image_path)
    train_count = 0
    for path, label in image_path:
        label_index = list(label).index('y')
        if train_count < 0.9 * n:
            shutil.copy(path, os.path.join(train_dir, task, 'train', str(label_index)))
        else:
            shutil.copy(path, os.path.join(train_dir, task, 'val', str(label_index)))
        train_count += 1
     
    
    
def get_model_net():
    pretrained_net = models.resnet50_v2(pretrained=True)
    
    num_gpu = 1
    ctx = [mx.gpu(i) for i in range(num_gpu)] if num_gpu > 0 else [mx.cpu()]
    
    finetune_net = models.resnet50_v2(classes=6)
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()
    return finetune_net, ctx



def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int), output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return((ap, cnt))

# 训练集图片增广（左右翻转，改颜色）
def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                   rand_crop=True, rand_mirror=True,
                                   mean = np.array([0.485, 0.456, 0.406]),
                                   std = np.array([0.229, 0.224, 0.225]))
    
    for aug in auglist:
        im - aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar())

# 验证集图片增广，没有随机裁剪和翻转
def transform_val(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                   mean = np.array([0.485, 0.456, 0.406]),
                                   std = np.array([0.229, 0.224, 0.225]))
    
    for aug in auglist:
        im - aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar())

# 在验证集上预测并评估
def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    AP = 0.
    AP_cnt = 0
    val_loss = 0
    for i, batch in anumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(x) for x in data]
        metric.update(label, outputs)
        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
        ap, cnt = calculate_ap(label, outputs)
        AP += ap
        AP_cnt += cnt
    _, val_acc = metric.get()
    return ((val_acc, AP / AP_cnt. val_loss / len(val_data)))
    
# =============================================================================
lr = 1e-3
momentum = 0.9
wd = 1e-4
epochs = 100
batch_size = 32

task = 'skirt_length_labels'
    
# 热身数据与训练数据的图片标记文件
warmup_label_dir = 'F://Data//03_FashionAI//warm//web//Annotations//skirt_length_labels.csv'
base_label_dir = 'F://Data//03_FashionAI//train//base//Annotations//label.csv'
train_data_dir = 'C://Soft//PythonWorkspace//FashionAIGame//train_valid'
image_path = []
    

if __name__ == '__main__':
    mkdir_if_not_exist(train_data_dir)
    
    
    with open(warmup_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, tk, label in tokens:
            image_path.append(('F://Data//03_FashionAI//warm//web//' + path , label))
    
    with open(base_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, tk, label in tokens:
            if tk == task:
                image_path.append(('F://Data//03_FashionAI//train//base//' + path , label))
    
    # 数据初始化，按照固定的目录结构存放
    # data_init(train_data_dir)
    
    train_path = os.path.join(train_data_dir, task, 'train')
    val_path = os.path.join(train_data_dir, task, 'val')
    
    # 定义训练集的 DataLoader （分批读取）
    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 定义验证集的 DataLoader
    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path, transform=transform_val),
        batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 获取迁移学习后的网络
    finetune_net, ctx = get_model_net()
    
    trainer = gluon.Trainer(finetune_net.collect_params(),
                           'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()


    for epoch in range(epochs):
        tic = time.time()
        
        train_loss = 0
        metric.reset()
        AP = 0.
        AP_cnt = 0
        
        num_batch = len(train_data)
        
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(x) for x in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
                
            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            
            metric.update(label, outputs)
            ap, cnt = calculate_ap(label, outputs)
            AP += ap
            AP_cnt += cnt
        
        train_map = AP / AP_cnt
        _, train_acc = metric.get()
        train_loss /= num_batch
        
        val_acc, val_map, val_loss = validate(finetune_net, val_data, ctx)
        print('[Epoch %d] Train-acc: %.3f, mAp: %.3f, loss: %.3f | val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.3f'%
             (epoch, train_acc, train_map, train_loss, val_acc, val_loss, time.time() - tic))
    