from abc import abstractmethod
from torch import nn
import torch
from .nn_utils import make_linear_model
from typing import Union, List


class Base(nn.Module):
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
        **kwargs
    ):
        super(FeedForward, self).__init__()
        self.output_dim = output_dim
        fc = make_linear_model(
            input_dim=input_dim,
            output_dim=output_dim,
            layers=layers,
            activation=activation,
            last_logit=last_logit,
            **kwargs
        )
        self.add_module("fc", fc)

    def __call__(self, x):
        return self.fc(x)

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
