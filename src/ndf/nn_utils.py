import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm as SN


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # torch.nn.init.xavier_uniform_(m.bias)
            torch.nn.init.zeros_(m.bias)


from typing import Union, List

# Num = Union[int, float]


def make_linear_model(
    input_dim: int, output_dim: int, layers: List[int], activation: str = "relu", last_logit: bool = False, **kwargs
):
    dropout_ratio = kwargs.get("dropout_ratio", 0.0)
    if dropout_ratio == 0.0:
        dropout_bool = False
    else:
        dropout_bool = True

    bn = kwargs.get("bn", None)
    if bn is None:
        bn_bool = False
    else:
        bn_bool = True
    if bn is not None:
        sn = False
    else:
        sn = kwargs.get("sn", False)
    bias = kwargs.get("bias", True)
    model = []
    act = map_act(activation)
    if sn & bn_bool:
        sn = (True,)
        bn_bool = False
    layers = [input_dim] + layers + [output_dim]
    for idx, layer in enumerate(layers[1:]):
        mod = nn.Linear(layers[idx], layer, bias=bias)
        model.append(mod)
        if (idx + 1) == len(layers[1:]):
            if last_logit:
                pass
            else:
                model.append(act())
        else:
            if sn:
                mod = SN(mod)
            if bn_bool:
                model.append(map_norm(bn, layer, **kwargs))
            model.append(act())
            if dropout_bool:
                model.append(nn.AlphaDropout(dropout_ratio))
    return nn.Sequential(*model).apply(weights_init)


def map_act(name):
    if name.lower() == "leaky_relu":
        activation = nn.LeakyReLU
    elif name.lower() == "mish":
        activation = Mish
    elif name.lower() == "selu":
        activation = nn.SELU
    elif name.lower() == "relu":
        activation = nn.ReLU
    elif name.lower() == "tanh":
        activation = nn.Tanh
    elif name.lower() == "gelu":
        activation = nn.GELU
    return activation


def map_norm(name, layer, **kwargs):
    """
    reference
    https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm
    1. batch_norm
    2. instance_norm
    3. group_norm
    4. layer_norm
    """
    group_n = int(kwargs.get("group_n", getmedianDivisors(layer)))
    if name.lower() == "batch_norm":
        norm = nn.BatchNorm1d(layer)
    elif name.lower() == "instance_norm":
        try:
            norm = nn.GroupNorm(int(layer / 2), num_channels=layer)
        except:
            norm = nn.GroupNorm(group_n, num_channels=layer)
    elif name.lower() == "group_norm":
        norm = nn.GroupNorm(group_n, num_channels=layer)
    elif name.lower() == "layer_norm":
        norm = nn.GroupNorm(1, num_channels=layer)
    else:
        norm = nn.BatchNorm1d(layer)
    return norm


def getmedianDivisors(n):
    import statistics

    r = []
    for i in np.arange(1, n):
        if n % i == 0:
            r.append(i)
    # max(set(r), key=r.count)
    return int(np.argsort(r)[len(r) // 2])
