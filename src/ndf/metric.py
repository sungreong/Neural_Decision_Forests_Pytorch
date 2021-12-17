import torchmetrics


class ClassificationMetric(object):
    def __init__(self, metric_kwargs={}):
        self.metrics = [
            torchmetrics.classification.F1(**metric_kwargs),
            #    torchmetrics.AUC(**metric_kwargs),
            torchmetrics.classification.Accuracy(**metric_kwargs),
            torchmetrics.classification.AveragePrecision(**metric_kwargs),
            torchmetrics.classification.AUROC(**metric_kwargs),
            torchmetrics.classification.FBeta(beta=0.5, **metric_kwargs),
        ]

        self.metric_dict = {type(metric).__name__: metric for metric in self.metrics}
        self.history = {k:[]  for k , v in self.metric_dict.items()}
        self.reset_log()

    def reset(
        self,
    ):
        for name, metric in self.metric_dict.items():
            metric.reset()

    def update(self, pred, target):
        for name, metric in self.metric_dict.items():
            metric.update(pred, target)

    def compute(self, to_numpy=True):
        result_dict = {}
        for name, metric in self.metric_dict.items():
            if to_numpy:
                result = metric.compute().detach().numpy()
            else:
                result = metric.compute()
            result_dict[name] = result
        return result_dict

    def log(self, result_dict):
        for name, metric in result_dict.items():
            self.history[name].append(metric)

    def reset_log(
        self,
    ):
        for _,value in self.history.items() :
            value.clear()