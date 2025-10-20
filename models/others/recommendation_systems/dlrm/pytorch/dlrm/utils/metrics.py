"""Customized implementation of metrics"""
import numpy as np
import torch

def ref_roc_auc_score(y_true, y_score, exact=True):
    """Compute AUC exactly the same as sklearn

    sklearn.metrics.roc_auc_score is a very genalized function that supports all kind of situation.
    See https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py. The AUC computation
    used by DLRM is a very small subset. This function is bear minimum codes of computing AUC exactly the same way
    as sklearn numerically.

    A lot of things are removed:
        Anything is not required by binary class.
        thresholds is not returned since we only need score.

    Args:
        y_true (ndarray or list of array):
        y_score (ndarray or list of array):
        exact (bool): If False, skip some computation used in sklearn. Default True
    """
    y_true = np.r_[y_true].flatten()
    y_score = np.r_[y_score].flatten()
    if y_true.shape != y_score.shape:
        raise TypeError(F"Shapre of y_true and y_score must match. Got {y_true.shape} and {y_score.shape}.")

    # sklearn label_binarize y_true which effectively make it integer
    y_true = y_true.astype(np.int)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    if exact:
        # Attempt to drop thresholds corresponding to points in between and collinear with other points.
        if len(fps) > 2:
            optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
            fps = fps[optimal_idxs]
            tps = tps[optimal_idxs]

    # Add an extra threshold position to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    direction = 1
    if exact:
        # I don't understand why it is needed since it is sorted before
        if np.any(np.diff(fpr) < 0):
            direction = -1

    area = direction * np.trapz(tpr, fpr)

    return area

def roc_auc_score(y_true, y_score):
    """Pytorch implementation almost follows sklearn

    Args:
        y_true (Tensor):
        y_score (Tensor):
    """
    device = y_true.device
    y_true.squeeze_()
    y_score.squeeze_()
    if y_true.shape != y_score.shape:
        raise TypeError(F"Shapre of y_true and y_score must match. Got {y_true.shape()} and {y_score.shape()}.")

    desc_score_indices = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = torch.nonzero(y_score[1:] - y_score[:-1], as_tuple=False).squeeze()
    threshold_idxs = torch.cat([distinct_value_indices, torch.tensor([y_true.numel() - 1], device=device)])

    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = torch.cat([torch.zeros(1, device=device), tps])
    fps = torch.cat([torch.zeros(1, device=device), fps])

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    area = torch.trapz(tpr, fpr)

    return area
