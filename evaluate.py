import sys

import joblib
import numpy as np
import rasterio
import os
import torch
import pandas as pd
from matplotlib import pyplot as plt

from utils.dataload import dataload_train
from torch.utils.data import DataLoader
from UNet import UNet
import tqdm

test_csv = './dataset/test_meta.csv'
test_csv = pd.read_csv(test_csv)

test_list = test_csv.to_numpy()
test_list[:, 0] = './dataset/test_img/' + test_list[:, 0]
# print(test_list)
# sys.exit()

test_dataset = dataload_train(path=test_list, H=256, W=256, aug=False, phase='test')
train_loader = DataLoader(dataset=test_dataset, batch_size=1)
device = torch.device('cuda:1')
sigmoid = torch.nn.Sigmoid()

if __name__ == '__main__':
    model = UNet(10, 1).to(device)
    model.load_state_dict(torch.load(r'D:\Spark_satellite\model\net_2.116522409778554e-06_E_40.pt'))
    model.eval()

    pbar = tqdm.tqdm(train_loader)

    pred_l = {}

    i = 0

    for _input in pbar:
        _input = _input.to(device)

        outputs = model(_input)
        segmentation_map = sigmoid(outputs)

        segmentation_map = segmentation_map.detach().cpu().numpy()
        # print(outputs.shape)
        # sys.exit()
        segmentation_map = np.where(segmentation_map[0, 0] > 0.8, 1, 0)
        pred = segmentation_map.astype(np.uint8)
        img_idx = str(test_list[i, 0]).rfind('/')
        pred_l[test_list[i, 0][img_idx+1:]] = pred

        if i % 200 == 0:
            plt.imshow(pred, cmap='gray')
            plt.show()
        i += 1

    print(pred_l)

    joblib.dump(pred_l, './y_pred.pkl')






