import argparse
import math
import os
from datetime import datetime, timedelta

import cv2 as cv
import numpy as np
from torch.nn import L1Loss

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import *


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def convert_matlab_datenum(matlab_datenum):
    part_1 = datetime.fromordinal(int(matlab_datenum))
    part_2 = timedelta(days=int(matlab_datenum) % 1)
    part_3 = timedelta(days=366)
    python_datetime = part_1 + part_2 - part_3
    return python_datetime


def num_years(begin, end_year):
    end = datetime(end_year, 6, 30)
    num_years = int((end - begin).days / 365.2425)
    return num_years


def get_sample(imdb, i):
    try:
        sample = dict()
        dob_list = imdb[0][0]
        dob = dob_list[i]
        dob = convert_matlab_datenum(dob)
        sample['dob'] = dob
        photo_taken_list = imdb[1][0]
        photo_taken = int(photo_taken_list[i])
        sample['photo_taken'] = photo_taken
        age = num_years(dob, photo_taken)
        sample['age'] = age
        full_path_list = imdb[2][0]
        full_path = os.path.join(IMG_DIR, full_path_list[i][0])
        sample['full_path'] = full_path
        gender_list = imdb[3][0]
        if math.isnan(gender_list[i]):
            gender = 2
        else:
            gender = int(gender_list[i])
        sample['gender'] = gender
        face_location_list = imdb[5][0]
        face_location = face_location_list[i][0]
        sample['face_location'] = face_location
        return sample
    except:
        pass
        # print('i: ' + str(i))
        # print('dob: ' + str(imdb[0][0][i]))
        # print('photo_taken: ' + str(imdb[1][0][i]))
        # print('full_path: ' + str(imdb[2][0][i][0]))
        # print('gender: ' + str(imdb[3][0][i]))
        # print('face_location: ' + str(imdb[5][0][i][0]))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_' + '.pth.tar'
    torch.save(state, filename)
    # 如果这个检查点是目前为止最好的，那么存储一个副本，这样它就不会被更差的检查点覆盖
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def mean_absolute_error(scores, targets):
    # print('scores.size(): ' + str(scores.size()))
    # print('targets.size(): ' + str(targets.size()))
    _, y_pred = scores.topk(1, 1, True, True)
    y_pred = y_pred.view(-1).float()
    y_true = targets.float()
    # print('y_pred.size(): ' + str(y_pred.size()))
    # print('y_true.size(): ' + str(y_true.size()))
    loss = L1Loss()
    return loss(y_pred, y_true)


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, cv.IMREAD_GRAYSCALE)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # 在裁剪设置中获取参考5地标位置
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def parse_args():
    """
    定义函数 parse_args,返回解析后的参数对象
    这将解析以下参数：
    - 使用预训练模型：`--pretrained True`
    - 使用 `resnet50` 网络架构：`--network resnet50`
    - 训练 100 个 epoch：`--end-epoch 100`
    - 初始学习率为 0.001：`--lr 0.001`
    - 批量大小为 256：`--batch-size 256`

    通过 `args` 对象，你可以在代码中访问这些参数的值。例如，`args.lr` 将是 `0.001`，`args.batch_size` 将是 `256`。

    ### 总结：
    `parse_args()` 函数是一个标准的命令行参数解析器，它定义了一些常见的训练配置参数，并返回一个包含解析结果的对象。使用这些参数，你可以灵活地调整模型训练过程中的设置。
    """
    parser = argparse.ArgumentParser(description='Train face network')                              # 创建 ArgumentParser 对象，description 提供了脚本的简短说明，在帮助信息中显示。此脚本用于训练人脸识别网络。
    # 一般
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')          # --pretrained 参数控制是否使用预训练模型。type=bool 表示它是一个布尔值，默认值为 False，帮助信息中会显示 "pretrained model"。
    parser.add_argument('--network', default='r50', help='specify network')                         # --network 参数用于指定使用的神经网络架构，默认为 'r50'，表示 ResNet-50。用户可以通过该参数选择其他架构。
    parser.add_argument('--end-epoch', type=int, default=50, help='training epoch size.')           # --end-epoch 参数用于指定训练的总轮次（epoch）。type=int 表示该参数为整数，默认值为 50，表示训练 50 个 epoch。
    parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')               # --lr 参数指定初始学习率。type=float 表示该参数是浮动类型（浮动的数字），默认值为 0.01。
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')    # --lr-step 参数用于指定学习率衰减的周期。每经过多少个 epoch，就会减小学习率。默认值为 10。
    parser.add_argument('--optimizer', default='sgd', help='optimizer')                             # --optimizer 参数用于指定训练过程中使用的优化器。默认为 'sgd'（随机梯度下降），但用户可以选择其他优化器（如 Adam）。
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')          # --weight-decay 参数用于控制权重衰减（L2 正则化）的强度，防止过拟合。默认值为 0.0005。
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')                          # --mom 参数指定优化器的动量。动量值通常在 0 到 1 之间，默认值为 0.9。
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')               # --emb-size 参数表示模型输出的人脸特征嵌入向量的长度。一般来说，这个值越大，模型能捕捉到的特征信息越多。默认值为 512。
    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')   # --batch-size 参数用于指定每次训练中所使用的批量大小。默认值为 512，即每次训练时使用 512 个样本。
    parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')                # --focal-loss 参数用于控制是否启用焦点损失（Focal Loss）。焦点损失是一种对类别不平衡有较强鲁棒性的损失函数，默认为 False。
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')        # --gamma 参数指定焦点损失函数中的聚焦参数。较大的值让模型更关注难以分类的样本，默认值为 2.0。
    parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')                   # --use-se 参数用于指定是否使用 SE Block（Squeeze-and-Excitation Block），SE Block 可以增强网络对不同通道的注意力。默认为 False。
    parser.add_argument('--age-weight', type=float, default=1.0, help='age loss weight')            # --age-weight 参数用于指定年龄损失（age loss）部分的权重。在多任务学习中，如果有年龄预测任务，可以通过调整此参数来平衡不同任务之间的权重。默认值为 1.0。
    args = parser.parse_args()                                                                      # 调用 parse_args() 方法解析命令行参数，将结果保存在 args 中
    return args                                                                                     # 返回解析后的参数对象 `args`，包含所有命令行传入的参数值
