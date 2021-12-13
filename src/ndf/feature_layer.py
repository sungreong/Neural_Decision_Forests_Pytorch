from abc import abstractmethod
from torch import nn
import torch
from .nn_utils import make_linear_model
from typing import Union, List
import numpy as np
from dataclasses import dataclass, field, asdict
import operator
from sklearn.preprocessing import LabelEncoder


class Base(nn.Module):
    def __init__(
        self,
    ):
        super(Base, self).__init__()

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    @abstractmethod
    def get_out_feature_size(self):
        raise NotImplementedError


class FeedForward(Base):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int],
        activation: str = "relu",
        last_logit: bool = True,
        **kwargs,
    ):
        super(FeedForward, self).__init__()
        self.output_dim = output_dim
        fc = make_linear_model(
            input_dim=input_dim,
            output_dim=output_dim,
            layers=layers,
            activation=activation,
            last_logit=last_logit,
            **kwargs,
        )
        self.add_module("fc", fc)

    def __call__(self, x):
        return self.fc(x)

    def get_out_feature_size(
        self,
    ):
        return self.output_dim


def embedding_layer(embedding_list: nn.ModuleList, x: np.ndarray):
    embedding_result = []
    for i , emb in enumerate(embedding_list):
        if isinstance(x, (np.ndarray,)):
            tensor = torch.LongTensor(x[:, int(i)])
        elif isinstance(x, (torch.Tensor,)):
            tensor = x[:, i].long()
        embed_v = emb(tensor)
        embedding_result.append(embed_v)
    else:
        return torch.cat(embedding_result, axis=1)


@dataclass(frozen=True)
class Category:
    position: int
    name: str
    unique_n: int
    unique_set: List[int] = field(default_factory=list)
    encoder: List[int] = field(default_factory=list)


def make_category_list(df, cat_cols):
    cat_list = []
    for idx, col in enumerate(cat_cols):
        
        encoder = LabelEncoder()
        encoder.fit(df[col])
        cat = Category(
            position=idx, name=col, unique_n=len(encoder.classes_), unique_set=encoder.classes_, encoder=encoder
        )
        cat_list.append(cat)
    else:
        pass
        #cat_list = sorted(cat_list, key=operator.attrgetter("position"))
    return cat_list


class CategoryEmbeddingLayer(nn.Module):
    def __init__(
        self,
        cat_list,
    ):
        super(CategoryEmbeddingLayer, self).__init__()
        self.module_list = torch.nn.ModuleList([])
                                           
        self.output_dim = 0
        for cat in cat_list:
            emb_dim = int(np.ceil(np.sqrt(cat.unique_n))) if cat.unique_n > 4 else cat.unique_n
            emb = nn.Embedding(cat.unique_n, emb_dim)
            if cat.unique_n > 4:
                pass
            else:
                emb.weight.data.fill_(1)
                emb.requires_grad_(requires_grad=False)
            self.module_list.append(emb)
            self.output_dim += emb_dim
            

            

    def __call__(self, x):
        return embedding_layer(self.module_list, x)

    def get_out_feature_size(
        self,
    ):
        return self.output_dim


# class NumericEmbeddingLayer(Base):
#     def __init__(
#         self,
#         cat_list,
#     ):
#         super(NumericEmbeddingLayer, self).__init__()
#         embedding_list = []
#         self.output_dim = 0
#         for cat in cat_list:
#             emb_dim = 4 if cat.unique_n > 4 else cat.unique_n
#             emb = nn.Embedding(cat.unique_n, emb_dim)
#             if cat.unique_n > 4:
#                 pass
#             else:
#                 emb.weight.data.fill_(1)
#                 emb.requires_grad_(requires_grad=False)
#             embedding_list.append([f"{cat.position}", emb])
#             self.output_dim += emb_dim

#         self.embedding_dict = nn.ModuleDict(embedding_list)

#     def __call__(self, x):
#         return embedding_layer(self.embedding_dict, x)

#     def get_out_feature_size(
#         self,
#     ):
#         return self.output_dim


@dataclass(frozen=True, order=True, unsafe_hash=True)
class Numeric:
    input_dim: int
    output_dim: int
    activation: str
    last_logit: bool
    layers: List[int] = field(default_factory=list)


class TabularLayer(Base):
    def __init__(self, num_info, cat_list=None):
        super(TabularLayer, self).__init__()

        self.num_emb_layer = make_linear_model(**asdict(num_info))
        self.cat_emb_layer = CategoryEmbeddingLayer(cat_list)
        self.output_dim = self.cat_emb_layer.output_dim + num_info.output_dim
        self.add_module("numeric", self.num_emb_layer)
        self.add_module("category", self.cat_emb_layer)

    def __call__(self, num_tensor, cat_tensor):
        num_emb_tensor = self.num_emb_layer(num_tensor)
        cat_emb_tensor = self.cat_emb_layer(cat_tensor)
        return torch.cat([num_emb_tensor, cat_emb_tensor], axis=1)

    def get_out_feature_size(
        self,
    ):
        return self.output_dim


if __name__ == "__main__":
    random_input = torch.randn(size=(100, 16))
    input_dim = random_input.shape[1]
    feature_layer = FeedForward(input_dim=16, output_dim=10, layers=[25, 25], activation="tanh", bn="batch_norm")
    print(feature_layer)
    print(feature_layer(random_input).size())
