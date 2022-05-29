# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import torch.nn as nn
import torch

def probability(pred, target, topk=1, thresh=None):
    """Calculate probability according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as probability. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """

    pred_value, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.squeeze(1)

    n_class = pred.shape[1]
    mask = (target >= 0) & (target < n_class)
    hist = torch.bincount(n_class * target[mask] + pred_label[mask], minlength=n_class ** 2).reshape(n_class, n_class).cpu().numpy()
    count = {}
    count = {i:c[i] for i,c in enumerate(hist)}
    prob = {
        k: v
        for k, v in sorted(
            count.items(), key=lambda item: item[1])
    }

    return prob


class Probability(nn.Module):
    """Probability calculation module."""

    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the probability.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                probability. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate probability.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return probability(pred, target, self.topk, self.thresh)
