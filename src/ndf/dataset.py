from numpy.lib.arraysetops import isin
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
        if isinstance(x,(pd.DataFrame,)) :
            self.x_numpy = x.values
        elif isinstance(x,(np.ndarray,)) :
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

# slow
class TabularBigDataSet(Dataset) :
    ignore_idx = 0
    
    def __init__(self, path, total_rows):
        self.path = path
        self.total_rows= total_rows
        self.total_idx_list = list(np.arange(total_rows))
        
        
        
    def __len__(self):
        return self.total_rows

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if idx == 0 :
            idx = int(np.random.choice(self.total_idx_list, size= 1 ))
            
        select_rows = [self.ignore_idx] + [idx]
            
        skip_rows = set(self.total_idx_list).difference(set(select_rows))
        one_row = pd.read_csv(self.path, skiprows=skip_rows)
        return torch.FloatTensor(one_row.values)
    
from io import StringIO
def load_with_buffer(filename, bool_skipped, **kwargs):
    s_buf = StringIO()
    with open(filename) as file:
        count = -1
        for line in file:
            count += 1
            if bool_skipped[count]:
                continue
            s_buf.write(line)
    s_buf.seek(0)
    df = pd.read_csv(s_buf, **kwargs)
    return df

# slow
class TabularBigDataSet_v2(Dataset) :
    
    def __init__(self, path, total_rows):
        self.path = path
        self.total_rows= total_rows
        self.total_idx_list = list(np.arange(total_rows))
        
        
        
    def __len__(self):
        return self.total_rows

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if idx == 0 :
            idx = int(np.random.choice(self.total_idx_list, size= 1 ))
            
        select_rows =  [idx]
            
        skip_rows = set(self.total_idx_list).difference(set(select_rows))
        skipped = np.asarray(list(skip_rows))
        # MAX >= number of rows in the file
        MAX = self.total_rows+1
        bool_skipped = np.zeros(MAX, dtype=bool)
        bool_skipped[skipped] = True
        one_row = load_with_buffer(self.path,bool_skipped,index_col=0)
        return torch.FloatTensor(one_row.values)
    

import random 

class IndexIter  :
    def __init__(self, idx_list , batch_size) :
        self.batch_size = batch_size
        idx_list.pop(0)
        self.idx_list = idx_list
        self.batch_count = len(idx_list) // batch_size
        self.remain_count = len(idx_list) -  ((len(idx_list) // batch_size) * batch_size)
        self._call_count = 0
        self.__iter__()
    
    def __next__(self,) :
        if self.batch_count <= self._call_count :
            raise StopIteration 
        select_idxs = self.chunks[self._call_count]
        self._call_count += 1
        return select_idxs

    def __iter__(self,) :
        random.shuffle(self.idx_list)
        self.chunks = [self.idx_list[i:i+self.batch_size-1] for i in range(0, len(self.idx_list) , self.batch_size)]
        self._call_count = 0 
        return self 
    

class TabularBigDataSet_v3(object) :
    ignore_idx = 0
    
    def __init__(self, path, total_rows, batch_size):
        self.path = path
        self.total_rows= total_rows
        self.batch_size = batch_size
        self.total_idx_list = list(np.arange(total_rows))
        self.iterator = IndexIter(self.total_idx_list, batch_size)
        
        
    def __len__(self):
        return self.total_rows
    
    def reset_used_index(self,) :
        self.used_index = []
        
    def get_candidate(self, ) :
        try :
            while True :
                return next(self.iterator)
                
        except StopIteration : 
            self.iterator = IndexIter(self.total_idx_list, self.batch_size)
            return next(self.iterator)
        
    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def batch(self, ):
        select_rows=  self.get_candidate()
        print(len(select_rows))
        skip_rows = set(self.total_idx_list).difference(set(select_rows))
        skipped = np.asarray(list(skip_rows))
        # MAX >= number of rows in the file
        MAX = self.total_rows+1
        bool_skipped = np.zeros(MAX, dtype=bool)
        bool_skipped[skipped] = True
        one_row = load_with_buffer(self.path,bool_skipped)
        print(one_row.head())
        print(one_row.shape)
        return torch.FloatTensor(one_row.values)