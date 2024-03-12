import os
import sys

import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import numpy as np
from numpy import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random

from net import *
import net as network

data_path = ""
batch_size = 4
H = 224
W = 224
dataloaders = {
    'train': DataLoader(dataload(path=r"D:\project_TransUNet\EVIMO\unet_test\train", H=H, W=W, pow_n=10, aug=True), batch_size=batch_size, shuffle=True, num_workers=5),
    'valid': DataLoader(dataload(path=r"D:\project_TransUNet\EVIMO\unet_test\valid", H=H, W=W, pow_n=10, aug=False), batch_size=batch_size, shuffle=False, num_workers=5)
}

from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


device_txt = "cuda:1"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_class = 1

if __name__ == '__main__':
    model = network.UNet(1, num_class).to(device)

    num_epochs = 2000
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    print("****************************GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    for epoch in range(num_epochs):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('------------------------' * 10)

        now = time.time()
        uu = ['train', 'valid'] if (epoch + 1) % 10 == 0 else ['train']

        for phase in uu:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)  # 성능 값 중첩
            epoch_samples = 0
            pbar = tqdm.tqdm(dataloaders[phase], unit='batch')
            # with tqdm.tqdm(dataloaders['train'], unit='batch', unit_scale=True) as data:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                # print(inputs.shape)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward computation
                    outputs, x5 = model(inputs)

                    # outputs = torch.sigmoid(outputs)
                    LOSS = L2_loss(outputs, labels)

                    metrics['Jointloss'] += LOSS

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)
            # print_metrics(metrics, epoch_samples, phase)

            if (epoch + 1) % 5 == 0:
                a = outputs[0].cpu().detach().numpy()
                x5 = x5.cpu().detach().numpy()
                # a = np.sum(a, axis=0)
                plt.imshow(a[0], cmap='gray')
                plt.show()
                plt.imshow(x5, cmap='gray')
                plt.show()

            pbar.close()

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            for param_group in optimizer.param_groups:
                lrrate = param_group['lr']
            print(phase, "Joint loss :", epoch_Jointloss.item(), 'lr rate', lrrate)
            # deep copy the model
            savepath = 'model/net_{}_E_{}.pt'
            if phase == 'valid' and epoch_Jointloss < best_loss:
                print("saving best model")
                best_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))

        print(time.time() - now)
