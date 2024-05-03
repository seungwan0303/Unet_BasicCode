import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import rasterio

import UNet as network
from utils.dataload import dataload_train
import pandas as pd
from torch import nn
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

random_state = 33
np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)

# def load_memory(path_mtx):
#     def load_tiff(path):
#         input = rasterio.open(path).read().transpose((1, 2, 0))
#         input = np.float32(input) / 65535
#
#         return input
#
#     for i in range(len(path_mtx)):
#         if i % 200 == 0:
#             print(i)
#         path_mtx[i, 0] = load_tiff(path_mtx[i, 0])
#         path_mtx[i, 1] = load_tiff(path_mtx[i, 1])
#
#     print('fin.')
#
#     return path_mtx

# valid_list = load_memory(valid_list)
# print(train_list)
# print(train_list[2])
# sys.exit()
# test_csv = pd.read_csv('./dataset/test_meta.csv')
# test_list = test_csv.to_numpy()
# test_list = './dataset/test_img/' + valid_list

H = 256
W = 256
batch_size = 16
num_workers = 0

def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 1

if __name__ == '__main__':
    train_csv = pd.read_csv('dataset/train_meta.csv')
    x_tr, x_val = train_test_split(train_csv, test_size=0.2, random_state=random_state)
    train_list = x_tr.to_numpy()
    train_list[:, 0], train_list[:, 1] = './dataset/train_img/' + train_list[:, 0], './dataset/train_mask/' + train_list[:, 1]
    # train_mtx = load_memory(train_list)
    # np.save('./train.npy', train_mtx)
    # del train_mtx
    valid_list = x_val.to_numpy()
    valid_list[:, 0], valid_list[:, 1] = './dataset/train_img/' + valid_list[:, 0], './dataset/train_mask/' + valid_list[:, 1]
    # valid_mtx = load_memory(valid_list)
    # np.save('./valid.npy', valid_mtx)
    # del valid_mtx
    # sys.exit()

    train_dataset = dataload_train(path=train_list, H=H, W=W, aug=True, phase='train')
    valid_dataset = dataload_train(path=valid_list, H=H, W=W, aug=False, phase='train')

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    model = network.UNet(10, num_classes).to(device)

    bce_criterion = nn.BCELoss()

    num_epochs = 1000
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    print("****************************GPU : ", device)
    sigmoid = nn.Sigmoid()

    best_loss = 1e10

    for epoch in range(1, num_epochs + 1):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('------------------------' * 10)

        phases = ['train', 'valid'] if epoch % 10 == 0 else ['train']

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0
            pbar = tqdm.tqdm(dataloaders[phase], unit='batch')

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = sigmoid(outputs)
                    LOSS = bce_criterion(outputs, labels)
                    metrics['Jointloss'] += LOSS

                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            if epoch % 10 == 0:
                pred = outputs[0].cpu().detach().numpy()
                plt.imshow(pred[0], cmap='gray')
                plt.show()

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples

            for param_group in optimizer.param_groups:
                lr_rate = param_group['lr']
            print(phase, "Joint loss :", epoch_Jointloss.item(), 'lr rate', lr_rate)

            savepath = 'model/net_{}_E_{}.pt'
            if phase == 'valid' and epoch_Jointloss < best_loss:
                print("model saved")
                best_loss = epoch_Jointloss
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))
