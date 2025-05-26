import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from PIL import Image
import os
import pandas as pd
from torch import nn
from datetime import datetime
import torch.nn.functional as F
import math


def log(data, a):  # loga(b)=ln(b)/ln(a)
    x_ones = torch.ones_like(data)
    a = x_ones * a
    data = torch.clamp(data, min=0)  # 把所有小于0的都变成0
    data = data + 1  # 相当于log（1+x）把所有数都挪到>0那边
    data = torch.log(data) / torch.log(a)
    return data

def show(a):
    plt.imshow(a)
    plt.show()

def to_np(a):
    return a.cpu().detach().numpy()

def get_data():
    root = '../data/KNMI/KNMI.h5'
    f = h5py.File(root, mode='r')
    train = f['train']
    test = f['test']
    train_image = train['images']
    test_image = test['images']
    return train_image, test_image


def psi(a, scale=4):
    # a shape is [B, S, H, W]
    B, S, H, W = a.shape
    C = scale ** 2
    new_H = int(H // scale)
    new_W = int(W // scale)
    a = np.reshape(a, (B, S, new_H, scale, new_W, scale))
    a = np.transpose(a, (0, 1, 3, 5, 2, 4))
    a = np.reshape(a, (B, S, C, new_H, new_W))
    return a

def inverse(a, scale=4):
    B, S, C, new_H, new_W = a.shape
    H = int(new_H * scale)
    W = int(new_W * scale)
    a = np.reshape(a, (B, S, scale, scale, new_H, new_W))
    a = np.transpose(a, (0, 1, 4, 2, 5, 3))
    a = np.reshape(a, (B, S, H, W))
    return a

def get_mask(eta, shape, test=False):
    B, S, C, H, W = shape
    if test:
        return torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0
    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1
    return eta, torch.tensor(mask, dtype=torch.float)

def data_2_rnn_mask(data, batch, batch_size, sequence, scale, eta, test=False):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)

    B, S, C, H, W = result.shape

    if test:
        return result, torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0

    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1

    return result, torch.tensor(mask, dtype=torch.float), eta


def data_2_rnn(data, batch, batch_size, sequence, scale):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)
    return result

    
def data_2_cnn(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)

    x = result[:, :9]
    y = result[:, 9:]
    return x, y
    

def inverse_cnn2(x, y):
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    x = to_np(x)
    y = to_np(y)
    x = inverse(x, scale=3)
    y = inverse(y, scale=3)

    x2 = np.zeros((x.shape[0], 9, 288, 288))
    y2 = np.zeros((y.shape[0], 9, 288, 288))

    index = 0

    for i in range(0, 864, 288):
        for j in range(0, 864, 288):
            x2[:, index] = x[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            y2[:, index] = y[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            index += 1
    return x2, y2


def _draw_color(t, flag, color):
    r = t[:, :, 0]
    g = t[:, :, 1]
    b = t[:, :, 2]
    r[flag] = color[0]
    g[flag] = color[1]
    b[flag] = color[2]
    return t


def draw_color_single(y):
    t = np.ones((y.shape[0], y.shape[1], 3)) * 255
    tt1 = []
    index = 0.5
    for i in range(30):
        tt1.append(index)
        index += 1
    color = [[28, 230, 180], [39, 238, 164], [58, 245, 143], [74, 248, 128], [97, 252, 108],
             [121, 254, 89], [143, 255, 73], [159, 253, 63], [173, 251, 56], [190, 244, 52],
             [203, 237, 52], [215, 229, 53], [227, 219, 56], [238, 207, 58], [246, 195, 58],
             [251, 184, 56], [254, 168, 51], [254, 153, 44], [253, 138, 38], [249, 120, 30],
             [244, 103, 23], [239, 88, 17], [231, 73, 12], [221, 61, 8], [212, 51, 5],
             [202, 42, 4], [188, 32, 2], [172, 23, 1], [158, 16, 1], [142, 10, 1]]

    for i in range(30):
        rain = y >= tt1[i]
        _draw_color(t, rain, color[i])
        
    t = t.astype(np.uint8)
    return t

def fundFlag(a, n, m):
    flag_1 = np.uint8(a >= n)
    flag_2 = np.uint8(a < m)
    flag_3 = flag_1 + flag_2
    return flag_3 == 2

def B_mse(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mse = np.sum(mask * ((a - b) ** 2)) / n
    return mse

def B_mae(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mae = np.sum(mask * np.abs(a - b)) / n
    return mae

def B_mse_SEVIR(a, b):
    mask = np.zeros(a.shape)
    mask[a < 0.7] = 1
    mask[fundFlag(a, 0.7, 3.5)] = 2
    mask[fundFlag(a, 3.5, 6.9)] = 5
    mask[fundFlag(a, 6.9, 12.0)] = 10
    mask[a > 12.0] = 30      
    n = a.shape[0] * b.shape[0]
    mse = np.sum(mask * ((a - b) ** 2)) / n
    return mse
        
def B_mae_SEVIR(a, b):
    mask = np.zeros(a.shape)
    mask[a < 0.7] = 1
    mask[fundFlag(a, 0.7, 3.5)] = 2
    mask[fundFlag(a, 3.5, 6.9)] = 5
    mask[fundFlag(a, 6.9, 12.0)] = 10
    mask[a > 12.0] = 30
    n = a.shape[0] * b.shape[0]
    mae = np.sum(mask * np.abs(a - b)) / n
    return mae

def R_mse(a, b):
    n = a.shape[0] * b.shape[0]
    mse = np.sum((a - b) ** 2) / n
    R_mse=np.sqrt(mse)
    return R_mse
    


def draw_color(data):
    B, C, H, W = data.shape
    result = torch.zoers((B, C, H, W, 3))
    for i in range(B):
        for j in range(C):
            result[B, C] = draw_color_single(data[B, C])
    return result

def tp(pre, gt):
    return np.sum(pre * gt)

def fn(pre, gt):
    a = pre + gt
    flag = (gt == 1) & (a == 1)
    return np.sum(flag)

def fp(pre, gt):
    a = pre + gt
    flag = (pre == 1) & (a == 1)
    return np.sum(flag)

def tn(pre, gt):
    a = pre + gt
    flag = a == 0
    return np.sum(flag)

def _csi(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    return TP / (TP + FN + FP + eps)


def _hss(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    a = TP * TN - FN * FP
    b = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + eps
    if a / b < 0:
        return 0
    return a / b

def csi(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_csi(a, b))
    return result

def hss(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_hss(a, b))
    return result

def csi_SEVIR(pred, gt):
    threshold = [0.14, 0.7, 3.5, 6.9, 12.0 ,32.0]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_csi(a, b))
    return result

def hss_SEVIR(pred, gt):
    threshold = [0.14, 0.7, 3.5, 6.9, 12.0 ,32.0]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_hss(a, b))
    return result

    
def data_Shanghai(x):
        
    x[x > 128] = 128
    print(torch.min(x),torch.max(x))  
    x=(x-0.0736)/128
    print(torch.min(x),torch.max(x))    
    
    return x

def inverse_data_Shanghai(x):
    x = x*128+0.0736
    print(torch.min(x),torch.max(x))
    return x

def inverse_transform_pixel(y):
    
    y=y*79.2614
    print(torch.min(y),torch.max(y))
    if torch.is_tensor(y) == False:
        y = torch.tensor(y, dtype=torch.float32)
    
    zero_condition = (y == 0)
    mid_condition = (y <= ((18 - 2) / 90.66))
    high_condition = (y > ((18 - 2) / 90.66))
    
    # 初始解设为None或者可以设为一个无效值
    x = torch.full_like(y, 0)
    # 当y == 0时
    x[zero_condition] = torch.tensor(0, dtype=torch.float32)  
    # 中间条件分支的解
    x[mid_condition] = 90.66 * (y[mid_condition]) + 2
    # 确保解在有效区间内
    x[(mid_condition) & ((x <= 5) | (x > 18))] = torch.tensor(0, dtype=torch.float32) 
    # 高条件分支的解
    x[high_condition] = 38.9 * torch.log(y[high_condition]) + 83.9
    # 确保解在有效区间内
    x[(high_condition) & ((x <= 18) | (x > 254))] = torch.tensor(0, dtype=torch.float32)

    return x


class BMAEloss(nn.Module):
    def __init__(self):
        super(BMAEloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()
        flag_2 = (a < m).int()
        flag_3 = flag_1 + flag_2
        return flag_3 == 2

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()
        mask[y < 2] = 1
        mask[self.fundFlag(y, 2, 5)] = 2
        mask[self.fundFlag(y, 5, 10)] = 5
        mask[self.fundFlag(y, 10, 30)] = 10
        mask[y > 30] = 30
        return torch.sum(mask * torch.abs(y - pred))
        
MAE_criterion = BMAEloss().cuda()
        
def CSM_Loss(inputs, target):
    losses = [MAE_criterion(inputs[i], target) for i in range(1,len(inputs))]
    mean_pooled_values = [F.avg_pool2d(inputs[i], kernel_size=(2, 2)).mean() for i in range(1,len(inputs))]
    scores = torch.cat([value.unsqueeze(0) for value in mean_pooled_values], dim=0)
    softmax_weights = F.softmax(scores, dim=0)
    
    losses_tensor = torch.stack(losses)
    weighted_loss=0
    for i in range(len(inputs)-1):
        weighted_loss+=softmax_weights[i] * losses_tensor[i]
   
    loss_one = MAE_criterion(inputs[0], target)
      
    return (weighted_loss + loss_one)/2 
    
            
class BMAEloss_SEVIR(nn.Module):
    def __init__(self):
        super(BMAEloss_SEVIR, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()
        flag_2 = (a < m).int()
        flag_3 = flag_1 + flag_2
        return flag_3 == 2

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()
        mask[y < 0.7] = 1
        mask[self.fundFlag(y, 0.7, 3.5)] = 2
        mask[self.fundFlag(y, 3.5, 6.9)] = 5
        mask[self.fundFlag(y, 6.9, 12.0)] = 10
        mask[y > 12.0] = 30
        return torch.sum(mask * torch.abs(y - pred))
        
MAE_criterion_SEVIR= BMAEloss_SEVIR().cuda()        

def CSM_Loss_SEVIR(inputs, target):
    losses = [MAE_criterion_SEVIR(inputs[i], target) for i in range(1,len(inputs))]
    mean_pooled_values = [F.avg_pool2d(inputs[i], kernel_size=(2, 2)).mean() for i in range(1,len(inputs))]
    scores = torch.cat([value.unsqueeze(0) for value in mean_pooled_values], dim=0)
    softmax_weights = F.softmax(scores, dim=0)
    
    losses_tensor = torch.stack(losses)
    weighted_loss=0
    for i in range(len(inputs)-1):
        weighted_loss+=softmax_weights[i] * losses_tensor[i]
   
    loss_one = MAE_criterion_SEVIR(inputs[0], target)
      
    return (weighted_loss + loss_one)/2 