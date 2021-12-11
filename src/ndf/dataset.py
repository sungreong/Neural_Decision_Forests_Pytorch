from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from .feature_layer import make_category_list
import pandas as pd
from typing import Union, List
from dataclasses import dataclass


class TabularNumDataset(Dataset):
    def __init__(self, x, y):
        self.x_numpy = x
        self.encoder = LabelEncoder()
        unique_label = set(y)
        try:
            unique_label.remove(np.nan)
        except KeyError as e:
            pass

        self.encoder.fit(list(unique_label))
        self.y_numpy = self.encoder.transform(y)
        if self.y_numpy.ndim == 1:
            self.y_numpy = self.y_numpy[:, None]

        self.target2idx = {idx: label for idx, label in enumerate(self.encoder.classes_)}
        self.idx2target = {v: k for k, v in self.target2idx.items()}

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_numpy)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_numpy[idx])
        y = torch.FloatTensor(self.y_numpy[idx])
        return x, y


class TabularNumCatDataset(Dataset):
    def __init__(self, x_num: pd.DataFrame, x_cat: pd.DataFrame, y: pd.DataFrame, cat_list: List[dataclass]):
        assert type(x_num) == pd.DataFrame
        assert type(x_cat) == pd.DataFrame
        assert type(y) == pd.DataFrame
        self.x_num_numpy = x_num.values
        self.x_cat_numpy = x_cat.values
        self.cat_list = cat_list

        self.encoder = LabelEncoder()
        unique_label = set(y.values.squeeze().tolist())
        try:
            unique_label.remove(np.nan)
        except KeyError as e:
            pass
        self.encoder.fit(list(unique_label))
        self.y_numpy = self.encoder.transform(y)
        if self.y_numpy.ndim == 1:
            self.y_numpy = self.y_numpy[:, None]

        self.target2idx = {idx: label for idx, label in enumerate(self.encoder.classes_)}
        self.idx2target = {v: k for k, v in self.target2idx.items()}

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_num_numpy)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x_num = torch.FloatTensor(self.x_num_numpy[idx])
        x_cat_list = []
        for cat in self.cat_list:
            x_cat_list.append(cat.encoder.transform(self.x_cat_numpy[idx][[cat.position]]))
        else:
            x_cat_np = np.array(x_cat_list).squeeze()
            # np.concatenate(x_cat_list, axis=1)
        x_cat = torch.LongTensor(x_cat_np)
        y = torch.FloatTensor(self.y_numpy[idx])
        return x_num, x_cat, y
