import numpy as np
from sklearn import metrics as metricser
from imblearn.metrics import geometric_mean_score


def evaluate_metrics(pred_proba, label):
    '''
    :param pred:
    :param label:
    :return:
    '''
    predicts = np.around(pred_proba)

    acc = metricser.accuracy_score(y_true=label, y_pred=predicts)
    f1 = metricser.f1_score(y_true=label, y_pred=predicts)
    auc = metricser.roc_auc_score(y_true=label, y_score=pred_proba)
    gmean = geometric_mean_score(y_true=label, y_pred=predicts)
    mcc = mcc_score(pred_proba, label)

    return acc, f1, auc, gmean, mcc


def mcc_score(pred_proba, label):
    trans_pred = np.ones(pred_proba.shape)
    trans_label = np.ones(label.shape)
    trans_pred[pred_proba < 0.5] = -1
    trans_label[label != 1] = -1
    mcc = metricser.matthews_corrcoef(trans_label, trans_pred)
    return mcc