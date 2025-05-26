import torch
import numpy as np
from torch import nn, optim
import matplotlib
import matplotlib.pyplot as plt
from RainHCNet import RainHCNet
import torch.nn.functional as F
from tqdm import tqdm, trange
import sys
sys.path.append('..')
from tool import *
from torch.utils.data import DataLoader
from skimage import measure
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# -----------------------------------------------
cuda_idx = 2
file_idx = 34
device = torch.device('cuda:0')
torch.cuda.set_device(device)
# -----------------------------------------------
test_seq = np.load('../test_seq.npy')
print(test_seq.shape[0])

train_data, test_data = get_data()
print("train_data:",train_data.shape)
print("test_data:",test_data.shape)

epoch_size, batch_size = 200, 26 
in_channel = 9
out_channel = 9
net = RainHCNet(in_channel, out_channel).cuda()

device = torch.device('cuda:0' if  torch.cuda.is_available() else 'cpu')
net.to(device)

min_test_loss, out_count = 1e10, 0
min_mae = 1e10
net.load_state_dict(torch.load('../PreTrainModel/model1_34_35.pt', map_location='cuda:0'))

# -----------------------------------------------

if file_idx == -1:
    f = open('log_rain_loss' + str(cuda_idx) +'_test'+ '.txt', 'a+')
else:
    f = open('log_' + str(file_idx) +'knmi_test'+ '.txt', 'a+')
# -----------------------------------------------
net.eval()

# 初始化列表
CSI, HSS, mse, mae = [], [], [], []
RainHCNet_CSI_KNMI = [[0] * 9 for _ in range(5)]
RainHCNet_HSS_KNMI = [[0] * 9 for _ in range(5)]
for i in range(5):
    CSI.append([])
    HSS.append([])
count=0
ran = np.arange(batch_size, test_seq.shape[0], batch_size)
pbar = tqdm(ran)
for batch in pbar:
    x, y = data_2_cnn(test_data, batch, batch_size, test_seq)
    x = x.to(device)
    y = y.to(device)
    y_hat = net(x)
    loss = CSM_Loss(y_hat, y)
    loss_num = loss.detach().cpu().numpy()
    pbar.set_description('Test MSE Loss: ' + str(loss_num / batch_size))    
    y_hat = to_np(y_hat[0])
    y = to_np(y)
    
    for i in range(batch_size):
        count+=1
        for j in range(9):
            a, b = y[i, j], y_hat[i, j]
            mse.append(B_mse(a, b))
            mae.append(B_mae(a, b))
            csi_result = csi(a, b)
            hss_result = hss(a, b)
            
            for t in range(5):
                CSI[t].append(csi_result[t])
                HSS[t].append(hss_result[t])
                RainHCNet_CSI_KNMI[t][j]+=(csi_result[t])
                RainHCNet_HSS_KNMI[t][j]+=(hss_result[t])
                
for i in range(5):
    CSI[i] = np.array(CSI[i]).mean()
    HSS[i] = np.array(HSS[i]).mean()
   
mse = np.array(mse).mean()
mae = np.array(mae).mean()


f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

threshold = [0.5, 2, 5, 10, 30]
print('CSI: ')
f.write('CSI: ')
print('CSI:')
for i in range(5):
    f.write('r>= ' + str(threshold[i]) + ' : ' + str(CSI[i]) + ' ')
    print('r>=', threshold[i], ':', CSI[i], end=' ')
    
f.write('\n')
print()
f.write('HSS: ')
print('HSS:')
for i in range(5):
    f.write('r>= ' + str(threshold[i]) + ' : ' + str(HSS[i]) + ' ')
    print('r>=', threshold[i], ':', HSS[i], end=' ')
f.write('\n')
print()

f.write('BMSE: ' + str(mse) + ' BMAE:' + str(mae) + '\n')
print('BMSE:', mse, 'BMAE:', mae)
seg_line = '=======================================================================' + '\n'
f.write(seg_line)
f.close()
