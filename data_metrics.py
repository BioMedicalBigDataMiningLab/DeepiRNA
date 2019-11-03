from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import numpy as np
import math
import win_unicode_console

win_unicode_console.enable()

aupr_average_list = []
auc_average_list = []
acc_average_list = []
f1_average_list = []
precision_average_list = []
recall_average_list = []
spec_average_list = []

# aupr, auc, f1, accuracy, recall, spec, precision
aupr_list = []
auc_list = []
f1_list = []
acc_list = []
precision_list = []
recall_list = []
spec_list = []


# Clear list
def clear_list():
    aupr_list.clear()
    auc_list.clear()
    f1_list.clear()
    acc_list.clear()
    precision_list.clear()
    recall_list.clear()
    spec_list.clear()


# The average result of list
def average_list(list):
    average = np.array(list).mean()
    return average


# Get list
def get_list(aupr, auc, f1, acc, recall, spec, precision):
    aupr_list.append(aupr)
    auc_list.append(auc)
    f1_list.append(f1)
    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    spec_list.append(spec)


# Add the every seed's avearage results to list
def get_average_results():
    auc_average_list.append(average_list(auc_list))
    acc_average_list.append(average_list(acc_list))
    f1_average_list.append(average_list(f1_list))
    aupr_average_list.append(average_list(aupr_list))
    spec_average_list.append(average_list(spec_list))
    precision_average_list.append(average_list(precision_list))
    recall_average_list.append(average_list(recall_list))


# Get the all seeds's results
def get_results():
    return auc_average_list, acc_average_list, f1_average_list, aupr_average_list, spec_average_list, precision_average_list, recall_average_list


# Calcaulate metrics
def get_metrics(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    recall = sen
    spec = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    max_index = np.argmax(f1)
    max_f1 = f1[0, max_index]
    max_accuracy = accuracy[0, max_index]
    max_recall = recall[0, max_index]
    max_spec = spec[0, max_index]
    max_precision = precision[0, max_index]
    return [max_f1, max_accuracy, max_recall, max_spec, max_precision]


# Evaluate results
def model_evaluate(real_score, predict_score):
    aupr = average_precision_score(real_score, predict_score)
    auc = roc_auc_score(real_score, predict_score)
    [f1, accuracy, recall, spec, precision] = get_metrics(real_score, predict_score)
    return np.array([aupr, auc, f1, accuracy, recall, spec, precision])
