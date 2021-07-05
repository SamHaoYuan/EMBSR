import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
"""
In this 
"""


# Precision
def one_precision(pre, truth, average='micro'):
    """
    根据输入的预测值和真实值，计算准确率; 这里我们考虑使用sklearn中计算多分类的方法; 适用于只有一个预测结果的情况
    :param pre: (B,1)
    :param truth: B,1
    :param average: 多分类加权方法
    :return:
    """
    precision = precision_score(truth, pre, average=average)
    return precision


# Recall
def one_recall(pre, truth, average='micro'):
    """
    根据输入的预测值和真实值，计算召回率; 这里我们考虑使用sklearn中计算多分类的方法; 适用于只有一个预测结果的情况
    :param pre:  B, 1
    :param truth: B, 1
    :param average: 多分类加权方法
    :return:
    """
    recall = recall_score(truth, pre, average=average)
    return recall


# F1
def one_f1(pre, truth, average='micro'):
    """
    根据输入的预测值和真实值，计算F-measure; 这里我们考虑使用sklearn中计算多分类的方法; 适用于只有一个预测结果的情况
    :param pre: B, 1
    :param truth: B, 1
    :param average: 多分类加权方法
    :return:
    """
    f1 = f1_score(truth, pre, average=average)
    return f1


# Recall
def get_recall(pre, truth):
    """
    根据输入的预测值和真实值，计算召回率
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """
    truths = truth.expand_as(pre)
    hits = (pre == truths).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    n_hits = (pre == truths).nonzero(as_tuple=False).size(0)
    recall = n_hits / truths.size(0)
    return recall
# F-measure


# MRR
def get_mrr(pre, truth):
    """
    计算MRR
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) TOP-K indics predicted by the model
    :return: MRR(Float), the mrr score
    """
    targets = truth.view(-1, 1).expand_as(pre)
    # ranks of the targets, if it appears in your indices
    hits = (targets == pre).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / targets.size(0)
    return mrr


# NDCG
def get_ndcg(pre, truth):
    """
    计算NDCG
    :param pre: （B,K)
    :param truth: (B, 1)
    :return:
    """
    targets = truth.view(-1, 1).expand_as(pre)  # B, K
    # ranks of the targets, if it appears in your indices
    hits = (targets == pre).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    dcg = 1/torch.log2(ranks + 1)  # 只有一个，不需要相加
    # 因为 target只有1，idcg可以简单算
    idcg = 1/np.log2(1+1)
    ndcg = torch.sum(dcg/idcg) / pre.size(0)
    return ndcg
