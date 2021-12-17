from torch import nn
import torch
from torch.nn.parameter import Parameter
from collections import OrderedDict
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from memory_profiler import profile 
# fp=open('NeuralDecisionForest.log','w+')

class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree

        return prob


class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training
        # used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        # leaf label distribution
        if jointly_training:
            self.pi = np.random.rand(self.n_leaf, n_class)
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=True)
        else:
            self.pi = np.ones((self.n_leaf, n_class)) / n_class
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        self.decision = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(n_used_feature, self.n_leaf)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x, self.feature_mask)  # ->[batch_size,n_used_feature]
        decision = self.decision(feats)  # ->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size, 1, 1).fill_(1.0))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            _mu = _mu * _decision  # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        return mu

    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi, dim=-1)
        else:
            return self.pi

    def cal_prob(self, mu, pi):
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi



class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, args):
        out = self.feature_layer(*args)
        out = out.view(args[0].size()[0], -1)
        out = self.forest(out)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    random_input = torch.randn(size=(100, 16))
    n_in_feature = random_input.shape[1]
    tree_params = dict(depth=3, n_in_feature=n_in_feature, used_feature_rate=0.5, n_class=2, jointly_training=True)
    tree = Tree(**tree_params)
    forest_param = dict(
        n_tree=5, tree_depth=3, n_in_feature=n_in_feature, tree_feature_rate=0.5, n_class=2, jointly_training=True
    )
    forest = Forest(**forest_param)
    print(tree(random_input).size())
    print(forest(random_input).size())

    from feature_layer import FeedForward

    feature_layer = FeedForward(
        input_dim=16, output_dim=10, layers=[25, 25], activation="tanh", bn="batch_norm", last_logit=True
    )
    forest_param = dict(
        n_tree=5, tree_depth=3, n_in_feature=10, tree_feature_rate=0.5, n_class=2, jointly_training=True
    )
    forest = Forest(**forest_param)
    nnForest = NeuralDecisionForest(feature_layer=feature_layer, forest=forest)
    print(nnForest(random_input).size())
    print([name for name, p in nnForest.named_parameters() if p.requires_grad])
