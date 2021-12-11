import torchmetrics
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from metric import ClassificationMetric
import numpy as np
import torch
from torch.utils.data import DataLoader


def train(model, optim, dataset, **kwargs):
    Loss_Collection = dict(train=[], test=[], loss=[])
    metric = torchmetrics.F1()
    metrics = ClassificationMetric(metric_kwargs=dict(num_classes=2))
    save_dir = kwargs.get("save_dir", "./")
    BEST_LOSS = np.inf
    for epoch in range(0, kwargs.get("n_epoch", 100)):
        model.train()
        trainloader = DataLoader(dataset, batch_size=kwargs.get("batch_size", 32), shuffle=kwargs.get("shuffle", True))
        Loss = []
        metric.reset()
        for batch_idx, (batch_x, batch_y) in enumerate(trainloader):
            output = model(batch_x)
            optim.zero_grad()
            loss = F.nll_loss(torch.log(output), batch_y.squeeze().long())
            loss.backward()
            optim.step()
            Loss.append(loss.detach().numpy())
            metric(output, batch_y.squeeze().long())
        else:
            f1 = metric.compute()
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
            metrics.reset()
            for batch_idx, (batch_x, batch_y) in enumerate(evalloader):
                output = model(batch_x)
                metric(output, batch_y.squeeze().long())
                metrics.update(output, batch_y.squeeze().long())
            else:
                result = metrics.compute()
                metrics.log(result)
                f1 = metric.compute()
            Loss_Collection["test"].append(f1 * 100)
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            axes = ax.flatten()
            logs_iter = [i * kwargs.get("n_log", 5) for i in range(len(Loss_Collection["test"]))]
            axes[0].plot(Loss_Collection["loss"])
            for metric_name, value in metrics.history.items():
                axes[1].plot(logs_iter, value, label=metric_name)
            # axes[1].plot(Loss_Collection["train"],label="train")
            # axes[1].plot(logs_iter,  Loss_Collection["test"],label = "test")
            axes[0].set_title("Loss")
            title = f"{type(metric).__name__} train best : {np.max(Loss_Collection['train']):.2f} test best : {np.max(Loss_Collection['test']):.2f}"
            axes[1].set_title("Test Metrics")
            plt.legend()
            plt.show()


# save_dir = Path("./../model")
# save_dir.mkdir(exist_ok=True)
# nnForest.load(save_dir.joinpath("./model.pt"))
# optim.load_state_dict(torch.load(save_dir.joinpath("./optim.pt")))
# train(nnForest , optim , dataset,batch_size=512,n_epoch=1000,save_dir=save_dir)
