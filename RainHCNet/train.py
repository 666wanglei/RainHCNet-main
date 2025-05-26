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
cuda_idx = 1
file_idx = 34
device = torch.device('cuda:0')
torch.cuda.set_device(device)

train_seq = np.load('../train_seq.npy')

val_seq = train_seq[5000:]
train_seq = train_seq[:5000]
print(len(train_seq))
print(len(val_seq))

train_data, test_data = get_data()
print("train_data.shape",train_data.shape)
print("test_data.shape",test_data.shape)
# -----------------------------------------------

epoch_size, batch_size = 100, 32
in_channel = 9
out_channel = 9
net = RainHCNet(in_channel, out_channel).cuda()

min_test_loss, out_count = 1e10, 0
min_mae = 1e10


# -----------------------------------------------
opt = optim.Adam(net.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, verbose=True)
MSE_criterion = BMAEloss().cuda()

# -----------------------------------------------
# 要画图的loss、csi、hss
train_epoch_loss_array = [0]
test_epoch_loss_array = [0]
csi_array = [[0],[0],[0],[0],[0]]
hss_array = [[0],[0],[0],[0],[0]]

for epoch in range(1, epoch_size + 1):
    if file_idx == -1:
        f = open('log_' + str(cuda_idx) + '.txt', 'a+')
    else:
        f = open('log_' + str(file_idx) + 'knmi.txt', 'a+')
    train_l_sum, test_l_sum, n = 0.0, 0.0, 0
    net.train()
    np.random.shuffle(train_seq)
    ran = np.arange(batch_size, train_seq.shape[0], batch_size)
    pbar = tqdm(ran)
    for batch in pbar:
        x, y = data_2_cnn(train_data, batch, batch_size, train_seq)
        x = x.cuda()
        y = y.cuda()
        y_hat = net(x)
        loss = CSM_Loss(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_num = loss.detach().cpu().numpy()
        pbar.set_description('Train MSE Loss: ' + str(loss_num / batch_size))
        train_l_sum += loss_num
        n += batch_size
    train_loss = train_l_sum / n
    train_epoch_loss_array.append(train_loss)
    n = 0
    net.eval()

    with torch.no_grad():
        np.random.shuffle(val_seq)
        ran = np.arange(batch_size, val_seq.shape[0], batch_size)
        pbar = tqdm(ran)
        CSI, HSS, mse, mae = [], [], [], []
        CSI_interval, HSS_interval, mse_interval, mae_interval = [], [], [], []
        for i in range(5):
            CSI.append([])
            HSS.append([])
            
        for batch in pbar:
            x, y = data_2_cnn(train_data, batch, batch_size, val_seq)
            x = x.cuda()
            y = y.cuda()
            y_hat = net(x)
            loss = CSM_Loss(y_hat, y)
            loss_num = loss.detach().cpu().numpy()
            test_l_sum += loss_num
            pbar.set_description('Test MSE Loss: ' + str(loss_num / batch_size))
            n += batch_size

            y = to_np(y)          
            y_hat = to_np(y_hat[0])
            for i in range(batch_size):
                for j in range(9):
                    a, b = y[i, j], y_hat[i, j]
                    mse.append(B_mse(a, b))
                    mae.append(B_mae(a, b))
                    csi_result = csi(a, b)
                    hss_result = hss(a, b)
                    for t in range(5):
                        CSI[t].append(csi_result[t])
                        HSS[t].append(hss_result[t])

        for i in range(5):
            CSI[i] = np.array(CSI[i]).mean()
            HSS[i] = np.array(HSS[i]).mean()
        mse = np.array(mse).mean()
        mae = np.array(mae).mean()

        f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        threshold = [0.5, 2, 5, 10, 30]
        f.write('CSI: ')
        print('CSI: ')
        for i in range(5):
            f.write('r>=' + str(threshold[i]) + ' : ' + str(CSI[i]) + ' ')
            csi_array[i].append(CSI[i])
            print('r>=', threshold[i], ':', CSI[i], end=' ')

        f.write('\n')
        print()
        f.write('HSS: ')
        print('HSS:')
        for i in range(5):
            f.write('r>=' + str(threshold[i]) + ' : ' + str(HSS[i]) + ' ')
            hss_array[i].append(HSS[i])
            print('r>=', threshold[i], ':', HSS[i], end=' ')
        f.write('\n')
        print()

        f.write('MSE: ' + str(mse) + ' MAE:' + str(mae) + '\n')
        print('MSE:', mse, 'MAE:', mae)

        test_loss = test_l_sum / n
        test_epoch_loss_array.append(test_loss)
        n = 0       
        
        lr_scheduler.step(test_loss)

        if epoch % 5 == 0:
            
            torch.save(net.state_dict(), 'model' + str(cuda_idx) + '_' + str(file_idx) + '_' + str(epoch) + '.pt')
            # 画图区
            matplotlib.use('Agg')
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            epochs = np.arange(1, epoch + 1)
            ax1.plot(epochs, train_epoch_loss_array[1:], 'r', label='Training Loss')
            ax1.plot(epochs, test_epoch_loss_array[1:], 'b', label='Validation Loss')
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('loss')
            ax1.set_title('Size Esimation loss vs. Training Epoch')
            ax1.legend() 
            plt.savefig("loss_knmi_RainHCNet.png")
            plt.close()

            plt.plot(np.arange(1, epoch + 1), csi_array[0][1:], 'y', label='0.5-2')
            plt.plot(np.arange(1, epoch + 1), csi_array[1][1:], 'r', label='2-5')
            plt.plot(np.arange(1, epoch + 1), csi_array[2][1:], color='orangered', label='5-10')
            plt.plot(np.arange(1, epoch + 1), csi_array[3][1:], color='blueviolet', label='10-30')
            plt.plot(np.arange(1, epoch + 1), csi_array[4][1:], color='green', label='>=30')
            plt.legend()  # 显示图例
            plt.xlabel('epochs')
            plt.ylabel('CSI')
            plt.savefig("CSI_knmi_RainHCNet.png")
            plt.close()

            plt.plot(np.arange(1, epoch + 1), hss_array[0][1:], 'y', label='0.5-2')
            plt.plot(np.arange(1, epoch + 1), hss_array[1][1:], 'r', label='2-5')
            plt.plot(np.arange(1, epoch + 1), hss_array[2][1:], color='orangered', label='5-10')
            plt.plot(np.arange(1, epoch + 1), hss_array[3][1:], color='blueviolet', label='10-30')
            plt.plot(np.arange(1, epoch + 1), hss_array[4][1:], color='green', label='>=30')
            plt.legend()  # 显示图例
            plt.xlabel('epochs')
            plt.ylabel('HSS')
            # plt.show()
            plt.savefig("HSS_knmi_RainHCNet.png")
            plt.close()
            
    f.write('Train loss: ' + str(train_loss) + ' Test loss: ' + str(test_loss) + '\n')
    seg_line = '=======================================================================' + '\n'
    f.write(seg_line)
    f.close()    
