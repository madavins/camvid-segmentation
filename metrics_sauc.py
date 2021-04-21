import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


eps = 1e-6

def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 4-d tensor in [batch_size, channels, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 4-d binarized y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data

def _argmax(y_data, dim):
    """
    args:
        y_data : 4-d tensor in [batch_size, chs, img_rows, img_cols]
        dim : int
    return 3-d [int] y_data
    """
    return torch.argmax(y_data, dim).int()


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : [int] 3-d in [batch_size, img_rows, img_cols]
        y_pred : [int] 3-d in [batch_size, img_rows, img_cols]
    return [float] true_positive
    """
    return torch.sum(y_true * y_pred).float()


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_positive
    """
    return torch.sum((1 - y_true) * y_pred).float()


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] true_negative
    """
    return torch.sum((1 - y_true) * (1 - y_pred)).float()


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_negative
    """
    return torch.sum(y_true * (1 - y_pred)).float()


def _get_weights(y_true, nb_ch):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        nb_ch : int 
    return [float] weights
    """
    batch_size, img_rows, img_cols = y_true.shape
    pixels = batch_size * img_rows * img_cols
    weights = [torch.sum(y_true==ch).item() / pixels for ch in range(nb_ch)]
    return weights

# modified by me # $
class CFMatrix(object):
    def __init__(self, des=None):
        self.des = des

    def __repr__(self):
        return "ConfusionMatrix"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, img_rows, img_cols] # $
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return confusion matrix
        """
        batch_size, chs, img_rows, img_cols = y_pred.shape # $
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = [nb_tp, nb_fp, nb_tn, nb_fn]
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            #y_true = _argmax(y_true, 1) # $
            performs = torch.zeros(chs, 4).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_tn = _get_tn(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch), :] = [nb_tp, nb_fp, nb_tn, nb_fn]
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs

# modified by me # $
class OAAcc(object):
    def __init__(self, des="Overall Accuracy"):
        self.des = des

    def __repr__(self):
        return "OAcc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, img_rows, img_cols] # $
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (tp+tn)/total
        """
        batch_size, chs, img_rows, img_cols = y_pred.shape # $
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
        else:
            y_pred = _argmax(y_pred, 1)
            #y_true = _argmax(y_true, 1) Els true_labels ja els tenim de [0..Nclasses-1]

        nb_tp_tn = torch.sum(y_true == y_pred).float()
        mperforms = nb_tp_tn / (batch_size * img_rows * img_cols)
        performs = None
        return mperforms, performs

# modified by me # $
class Precision(object):
    def __init__(self, des="Precision"):
        self.des = des

    def __repr__(self):
        return "Prec"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, img_rows, img_cols] # $
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fp)
        """
        batch_size, chs, img_rows, img_cols = y_pred.shape # $
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fp + eps)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            #y_true = _argmax(y_true, 1) # $
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fp + eps)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs

# modified by me # $
class Recall(object):
    def __init__(self, des="Recall"):
        self.des = des

    def __repr__(self):
        return "Reca"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, img_rows, img_cols] # $
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fn)
        """
        batch_size, chs, img_rows, img_cols = y_pred.shape # $
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fn + eps)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            #y_true = _argmax(y_true, 1) # $
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fn + eps)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs

# modified by me # $
class F1Score(object):
    def __init__(self, des="F1Score"):
        self.des = des

    def __repr__(self):
        return "F1Sc"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, img_rows, img_cols] # $
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols] 
            threshold : [0.0, 1.0]
        return 2*precision*recall/(precision+recall)
        """
        batch_size, chs, img_rows, img_cols = y_pred.shape # $
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            _precision = nb_tp / (nb_tp + nb_fp + eps)
            _recall = nb_tp / (nb_tp + nb_fn + eps)
            mperforms = 2 * _precision * _recall / (_precision + _recall + eps)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            #y_true = _argmax(y_true, 1) 
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                _precision = nb_tp / (nb_tp + nb_fp + eps)
                _recall = nb_tp / (nb_tp + nb_fn + eps)
                performs[int(ch)] = 2 * _precision * \
                    _recall / (_precision + _recall + eps)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs

# modified by me # $
class Kappa(object):
    def __init__(self, des="Kappa"):
        self.des = des

    def __repr__(self):
        return "Kapp"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, img_rows, img_cols] # $
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (Po-Pe)/(1-Pe)
        """
        batch_size, chs, img_rows, img_cols = y_pred.shape # $
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            nb_total = nb_tp + nb_fp + nb_tn + nb_fn
            Po = (nb_tp + nb_tn) / nb_total
            Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn) +
                  (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
            mperforms = (Po - Pe) / (1 - Pe + eps)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            #y_true = _argmax(y_true, 1) # $
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_tn = _get_tn(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                nb_total = nb_tp + nb_fp + nb_tn + nb_fn
                Po = (nb_tp + nb_tn) / nb_total
                Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn)
                      + (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
                performs[int(ch)] = (Po - Pe) / (1 - Pe + eps)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs

# modified by me # $
class Jaccard(object):
    def __init__(self, des="Jaccard"):
        self.des = des

    def __repr__(self):
        return "Jacc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, img_rows, img_cols] # $
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols] 
            threshold : [0.0, 1.0]
        return intersection / (sum-intersection)
        """
        batch_size, chs, img_rows, img_cols = y_pred.shape # $
        device = y_true.device
        
        y_pred = _argmax(y_pred, 1)
        #y_true = _argmax(y_true, 1) Els true_labels ja els tenim de [0..Nclasses-1]
        performs = torch.zeros(chs, 1).to(device)
        weights = _get_weights(y_true, chs)
        
        for ch in range(chs):
            y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
            y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
            y_true_ch[y_true == ch] = 1
            y_pred_ch[y_pred == ch] = 1
            _intersec = torch.sum(y_true_ch * y_pred_ch).float()
            _sum = torch.sum(y_true_ch + y_pred_ch).float()
            
            performs[int(ch)] = _intersec / (_sum - _intersec + eps)
            
        mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        miou = torch.mean(performs, dim=0)
        return mperforms, miou, performs









