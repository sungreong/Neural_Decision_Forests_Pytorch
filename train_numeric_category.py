import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from ndf import make_category_list
from ndf import TabularNumCatDataset
from torch.utils.data import DataLoader
from ndf import Numeric, TabularLayer
from ndf import NeuralDecisionForest, Forest
from ndf import ClassificationMetric
import torchmetrics
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_dir = Path("./data")
df = pd.read_csv(data_dir.joinpath("./adult_train.csv"))
test = pd.read_csv(data_dir.joinpath("./adult_test.csv"))
test


num_cols = list(df.select_dtypes("number"))
target_col = ["income_bracket"]


num_cols = list(df.select_dtypes("number"))
target_col = ["income_bracket"]
cat_cols = list(set(list(df.columns)).difference(set(num_cols + target_col)))
total_cols = num_cols + target_col
data = df[total_cols]
y = df[target_col]
numX = df[num_cols]
catX = df[cat_cols]


testNumX = test[num_cols]
testCatX = test[cat_cols]
testY = test[target_col]


cat_cols = list(catX)
cat_list = make_category_list(catX, cat_cols)


num_info = Numeric(input_dim=numX.shape[1], output_dim=10, activation="relu", last_logit=True, layers=[25])
tabularLayer = TabularLayer(num_info, cat_list)


feature_layer = tabularLayer
feature_dim = tabularLayer.output_dim
forest_param = dict(
    n_tree=5, tree_depth=3, n_in_feature=feature_dim, tree_feature_rate=0.5, n_class=2, jointly_training=True
)
forest = Forest(**forest_param)
nnForest = NeuralDecisionForest(feature_layer=feature_layer, forest=forest)


def prepare_optim(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=1e-5)


dataset = TabularNumCatDataset(numX, catX, y, cat_list)
testdataset = TabularNumCatDataset(testNumX, testCatX, testY, cat_list)

optim = prepare_optim(nnForest, 1e-5)


def train(model, optim, dataset, testdataset, **kwargs):
    Loss_Collection = dict(train=[], test=[], loss=[])
    metric = torchmetrics.F1()
    train_metrics = ClassificationMetric(metric_kwargs=dict(num_classes=2))
    test_metrics = ClassificationMetric(metric_kwargs=dict(num_classes=2))
    save_dir = kwargs.get("save_dir", "./")
    BEST_LOSS = np.inf
    for epoch in range(0, kwargs.get("n_epoch", 100)):
        model.train()
        trainloader = DataLoader(dataset, batch_size=kwargs.get("batch_size", 32), shuffle=kwargs.get("shuffle", True))
        Loss = []
        metric.reset()
        for batch_idx, (*batch_x, batch_y) in enumerate(trainloader):
            output = model(batch_x)
            optim.zero_grad()
            loss = F.nll_loss(torch.log(output), batch_y.squeeze().long())
            loss.backward()
            optim.step()
            Loss.append(loss.detach().numpy())
            metric(output, batch_y.squeeze().long())
            train_metrics.update(output, batch_y.squeeze().long())
        else:
            f1 = metric.compute()
            result = train_metrics.compute()
            train_metrics.log(result)
            Loss_Collection["loss"].append(np.mean(Loss))
            Loss_Collection["train"].append(f1 * 100)
            if BEST_LOSS > np.mean(Loss):
                BEST_LOSS = np.mean(Loss)
                model.save(save_dir.joinpath("./model.pt"))
                torch.save(optim.state_dict(), save_dir.joinpath("./optim.pt"))

        if epoch % kwargs.get("n_log", 5) == 0:
            clear_output(wait=True)
            evalloader = DataLoader(dataset, batch_size=kwargs.get("batch_size", 32), shuffle=False)

            metric.reset()
            model.eval()
            test_metrics.reset()
            for batch_idx, (*batch_x, batch_y) in enumerate(evalloader):
                output = model(batch_x)
                metric(output, batch_y.squeeze().long())
                test_metrics.update(output, batch_y.squeeze().long())
            else:
                result = test_metrics.compute()
                test_metrics.log(result)
                f1 = metric.compute()
            Loss_Collection["test"].append(f1 * 100)
            total_length = 1 + len(test_metrics.history)
            figsize = 5
            nROWS = 3
            nCols = int(np.ceil(total_length / nROWS))
            fig, ax = plt.subplots(nROWS, nCols, figsize=(nROWS * figsize, nCols * figsize))
            plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05, right=0.99, left=0.05)
            axes = ax.flatten()
            logs_iter = [i * kwargs.get("n_log", 5) for i in range(len(Loss_Collection["test"]))]
            axes[0].plot(Loss_Collection["loss"], label="Train Loss")
            for idx, key in enumerate(list(test_metrics.history.keys())):
                figure_idx = idx + 1
                te_metric_result = test_metrics.history[key]
                tr_metric_result = train_metrics.history[key]

                axes[figure_idx].plot(tr_metric_result, label=f"train : {key}")
                axes[figure_idx].plot(logs_iter, te_metric_result, label=f"test : {key}")
                axes[figure_idx].set_title(key)
                axes[figure_idx].legend(loc="upper right")
            axes[0].set_title("Loss")
            axes[0].legend()

            if kwargs.get("save_fig", None) is not None:
                plt.savefig(save_dir.joinpath(kwargs.get("save_fig", None)))
                plt.close()
            else:
                plt.show()


save_dir = Path("./model_v2")
save_dir.mkdir(exist_ok=True)
# nnForest.load(save_dir.joinpath("./model.pt"))
# optim.load_state_dict(torch.load(save_dir.joinpath("./optim.pt")))
train(
    nnForest,
    optim,
    dataset,
    testdataset,
    batch_size=512,
    n_epoch=100000,
    save_dir=save_dir,
    save_fig="./monitor.png",
)
