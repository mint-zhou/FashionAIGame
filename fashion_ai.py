import os
import sys
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


# 解析图片标记文件中的路径，获取其中的路径
def get_all_image_path(base_label_dir, base_pic_dir, task):
    all_image_path = []

    with open(base_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, tk, label in tokens:
            if tk == task:
                all_image_path.append((base_pic_dir + path, label))

    print('[get_all_image_path] [task]' + task + '; number: ' + str(len(all_image_path)))
    return all_image_path


# 拷贝图片，以特定的目录结构保存
def copy_all_image(image_path, task, rate):
    m = len(list(image_path[0][1]))
    for mm in range(m):
        mkdir_if_not_exist([train_data_dir, task, 'train', str(mm)])
        mkdir_if_not_exist([train_data_dir, task, 'val', str(mm)])

    n = len(image_path)
    random.seed(1024)
    random.shuffle(image_path)
    train_count = 0
    for path, label in image_path:
        label_index = list(label).index('y')
        if train_count < rate * n:
            shutil.copy(path, os.path.join(train_data_dir, task, 'train', str(label_index)))
        else:
            shutil.copy(path, os.path.join(train_data_dir, task, 'val', str(label_index)))
        train_count += 1

    print('[copy_all_picture] [task]' + task + '; number: ' + str(train_count))


# 图片数据预处理与初始化，保存成特定的目录结构
def data_preprocess(base_label_dir, base_pic_dir, train_data_dir, task_list):
    all_image_path = []
    mkdir_if_not_exist(train_data_dir)

    for task in task_list:
        print('[data_preprocess] [task]', task)

        # 解析图片标记文件中的路径，获取其中的路径
        all_image_path = get_all_image_path(base_label_dir, base_pic_dir, task)

        mkdir_if_not_exist([train_data_dir, task])
        mkdir_if_not_exist([train_data_dir, task, 'train'])
        mkdir_if_not_exist([train_data_dir, task, 'val'])

        # 拷贝图片，以特定的目录结构保存
        copy_all_image(all_image_path, task, 0.9)


# =================================================================================================================
def get_gpu(num_gpu):
    if num_gpu > 0:
        ctx = [mx.gpu(i) for i in range(num_gpu)]
    else:
        ctx = [mx.cpu()]
    return ctx


# 获取resnet34_v2模型，并微调（迁移学习）
def get_model_resnet34_v2(classes_num, ctx):
    pretrained_net = models.resnet34_v2(pretrained=True)
    # print(pretrained_net)

    finetune_net = models.resnet34_v2(classes=classes_num)      # 输出为classes_num个类
    finetune_net.features = pretrained_net.features             # 特征设置为resnet34_v2的特征
    finetune_net.output.initialize(init.Xavier(), ctx=ctx)      # 对输出层做初始化
    finetune_net.collect_params().reset_ctx(ctx)                # 设置CPU或GPU
    finetune_net.hybridize()                                    # gluon特征，运算转成符号运算，提高运行速度
    return finetune_net


# 获取resnet50_v2模型，并微调（迁移学习）
def get_model_resnet50_v2(classes_num, ctx):
    pretrained_net = models.resnet50_v2(pretrained=True)
    # print(pretrained_net)

    finetune_net = models.resnet50_v2(classes=classes_num)      # 输出为classes_num个类
    finetune_net.features = pretrained_net.features             # 特征设置为resnet50_v2的特征
    finetune_net.output.initialize(init.Xavier(), ctx=ctx)      # 对输出层做初始化
    finetune_net.collect_params().reset_ctx(ctx)                # 设置CPU或GPU
    finetune_net.hybridize()                                    # gluon特征，运算转成符号运算，提高运行速度
    return finetune_net


# 获取inception_v3模型，并微调（迁移学习）
def get_model_inception_v3(classes_num, ctx):
    pretrained_net = models.inception_v3(pretrained=True)
    # print(pretrained_net)

    finetune_net = models.inception_v3(classes=classes_num)     # 输出为classes_num个类
    finetune_net.features = pretrained_net.features             # 特征设置为inceptionv3的特征
    finetune_net.output.initialize(init.Xavier(), ctx=ctx)      # 对输出层做初始化
    finetune_net.collect_params().reset_ctx(ctx)                # 设置CPU或GPU
    finetune_net.hybridize()                                    # gluon特征，运算转成符号运算，提高运行速度
    return finetune_net


# 获取alexnet模型，并微调（迁移学习）
def get_model_alexnet(classes_num, ctx):
    pretrained_net = models.alexnet(pretrained=True)
    # print(pretrained_net)

    finetune_net = models.alexnet(classes=classes_num)          # 输出为classes_num个类
    finetune_net.features = pretrained_net.features             # 特征设置为inceptionv3的特征
    finetune_net.output.initialize(init.Xavier(), ctx=ctx)      # 对输出层做初始化
    finetune_net.collect_params().reset_ctx(ctx)                # 设置CPU或GPU
    finetune_net.hybridize()                                    # gluon特征，运算转成符号运算，提高运行速度
    return finetune_net


# 获取vgg19模型，并微调（迁移学习）
def get_model_vgg19(classes_num, ctx):
    pretrained_net = models.vgg19(pretrained=True)
    # print(pretrained_net)

    finetune_net = models.vgg19(classes=classes_num)          # 输出为classes_num个类
    finetune_net.features = pretrained_net.features             # 特征设置为inceptionv3的特征
    finetune_net.output.initialize(init.Xavier(), ctx=ctx)      # 对输出层做初始化
    finetune_net.collect_params().reset_ctx(ctx)                # 设置CPU或GPU
    finetune_net.hybridize()                                    # gluon特征，运算转成符号运算，提高运行速度
    return finetune_net


# 为模型添加dropout，加在倒二层
def add_model_dropout(old_net, layers_count, dropout):
    new_net = nn.HybridSequential()
    for i in range(layers_count):
        if i is (layers_count - 1):
            new_net.add(nn.Dropout(dropout))
        new_net.add(old_net.features[i])

    return new_net


# 为模型添加dropout，加在倒二层
def del_model_dropout(old_net, layers_count):
    new_net = nn.HybridSequential()
    for i in range(layers_count):
        if i is not (layers_count - 1):
            new_net.add(old_net.features[i])

    return new_net

# =================================================================================================================
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
    aug_list = image.CreateAugmenter(data_shape=(3, 299, 299), resize=256,
                                     rand_crop=True, rand_mirror=True,
                                     mean=np.array([0.485, 0.456, 0.406]),
                                     std=np.array([0.229, 0.224, 0.225]))

    for aug in aug_list:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar())


# 验证集图片增广，没有随机裁剪和翻转
def transform_val(data, label):
    im = data.astype('float32') / 255
    aug_list = image.CreateAugmenter(data_shape=(3, 299, 299), resize=256,
                                   mean=np.array([0.485, 0.456, 0.406]),
                                   std=np.array([0.229, 0.224, 0.225]))
    
    for aug in aug_list:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar())


# 在验证集上预测并评估
def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    AP = 0.
    AP_cnt = 0
    val_loss = 0
    for i, batch in enumerate(val_data):
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
    return ((val_acc, AP / AP_cnt, val_loss / len(val_data)))


# 开始训练
def start_train(train_data_dir, save_model_dir, task, epochs, batch_size, classes_num, dropout, lr, momentum, wd):
    print("[start_train] [task] " + task + " start")
    mkdir_if_not_exist(save_model_dir)
    save_model_name = os.path.join(save_model_dir, task + ".params")
    print('[save_model_name]', save_model_name)

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

    ctx = get_gpu(1)
    # 获取迁移学习后的网络
    finetune_net = get_model_inception_v3(classes_num=classes_num, ctx=ctx)
    # finetune_net = get_model_resnet34_v2(classes_num=classes_num, ctx=ctx)
    finetune_net.load_params(filename=save_model_name, ctx=ctx)
    # finetune_net = add_model_dropout(old_net=finetune_net, layers_count=len(finetune_net.features), dropout=dropout)

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
        print('[Epoch %d] Train-acc: %.3f, mAp: %.3f, loss: %.3f | val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.3f' %
              (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))

    finetune_net.save_params(save_model_name)


# =================================================================================================================
lr = 1e-3
momentum = 0.9
wd = 1e-4
epochs = 50
batch_size = 8
dropout = 0.6

# 热身数据与训练数据的图片标记文件
base_label_dir = 'F://Data//03_FashionAI//train//base//Annotations//label.csv'
base_pic_dir = 'F://Data//03_FashionAI//train//base//'
train_data_dir = os.path.join(sys.path[0], 'train_valid')
save_model_dir = os.path.join(sys.path[0], 'train_models')

task_list = [('skirt_length_labels', 6),
             ('coat_length_labels', 8),
             ('collar_design_labels', 5),
             ('lapel_design_labels', 5),
             ('neck_design_labels', 5),
             ('neckline_design_labels', 10),
             ('pant_length_labels', 6),
             ('sleeve_length_labels', 9)]


if __name__ == '__main__':
    # 图片数据预处理与初始化，保存成特定的目录结构
    # data_preprocess(base_label_dir, base_pic_dir, train_data_dir, task_list)

    for task, classes_num in task_list:
        start_train(train_data_dir, save_model_dir, task, epochs, batch_size, classes_num, dropout, lr, momentum, wd)

    # ctx = get_gpu(1)
    # my_net = get_model_resnet34_v2(6, ctx)
    # get_model_vgg19(6, ctx)
    # print(my_net)
    # my_net = add_model_dropout(my_net, len(my_net.features), 0.5)
    # print(my_net)
    # my_net = del_model_dropout(my_net, 14)
    # print(my_net)
