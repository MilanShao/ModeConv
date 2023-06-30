import pickle
import random
import string

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import mahalanobis
import datetime
import time
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from scipy import linalg


def generate_date_prefix(random_letters=True) -> str:
    out = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}-{datetime.datetime.now().minute}'
    if random_letters:
        t = 1000 * time.time()
        random.seed(int(t) % 2 ** 32)
        out = f'{out}_{"".join(random.choices(string.ascii_uppercase, k=5))}'
    return out


def compute_maha_threshold(val_labels, val_logits):
    thresholds = []
    precisions = []
    recalls = []
    f1s = []
    bal_accs = []

    ids = np.where(val_labels == 0)

    val_logits_normal = val_logits.reshape((len(val_labels), -1))[ids]
    val_logits = val_logits.reshape((len(val_labels), -1))
    mean = val_logits_normal.mean(0)
    cov = np.cov(val_logits_normal.T, val_logits_normal.T)[:val_logits_normal.shape[-1], :val_logits_normal.shape[-1]]
    cov = linalg.pinv(cov)
    maha_dist = np.array([mahalanobis(x, mean, cov) for x in val_logits_normal])
    all_maha_dist = np.array([mahalanobis(x, mean, cov) for x in val_logits])
    for i in range(10):
        maha_thresh = np.quantile(maha_dist, 0.9 + i / 100)
        predictions = np.where(all_maha_dist > maha_thresh, True, False)

        tn, fp, fn, tp = confusion_matrix(val_labels, predictions, labels=[0, 1]).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        bal_acc = balanced_accuracy_score(val_labels, predictions)

        thresholds.append(maha_thresh)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        bal_accs.append(bal_acc)

    val_df = pd.DataFrame(np.vstack((thresholds, precisions, recalls, f1s, bal_accs)).T,
                          columns=["Threshold", "Precision", "Recall", "F1", "bal_acc"])

    return val_df, mean, cov


def compute_thresholds(val_losses, val_labels):
    ids = np.where(val_labels == 0)
    max_val_loss = val_losses[ids[0]].max()

    thresholds = []
    precisions = []
    recalls = []
    f1s = []
    bal_accs = []

    for threshold in (np.arange(0, max_val_loss * 2, max_val_loss * 2 / 200)):
        predictions = np.where(val_losses > threshold, True, False)

        tn, fp, fn, tp = confusion_matrix(val_labels, predictions, labels=[0, 1]).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        bal_acc = balanced_accuracy_score(val_labels, predictions)

        thresholds.append(threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        bal_accs.append(bal_acc)

    thresholds = np.array(thresholds)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    bal_accs = np.array(bal_accs)

    val_df = pd.DataFrame(np.vstack((thresholds, precisions, recalls, f1s, bal_accs)).T,
                          columns=["Threshold", "Precision", "Recall", "F1", "bal_acc"])

    return val_df, max_val_loss


def compute_and_save_metrics(model):
    model.test_scores = torch.hstack(model.test_scores)
    model.test_labels = torch.hstack(model.test_labels)
    test_scores = model.test_scores.cpu().detach().numpy()
    test_labels = model.test_labels.cpu().detach().numpy()

    predictions = np.where(test_scores > model.best_threshold, True, False)

    auc = roc_auc_score(test_labels, test_scores)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    bal_acc = balanced_accuracy_score(test_labels, predictions)

    # predictions2 = np.where(test_scores > model.max_val_loss, True, False)

    # tn2, fp2, fn2, tp2 = confusion_matrix(test_labels, predictions2).ravel()
    # precision2 = tp2 / (tp2 + fp2)
    # recall2 = tp2 / (tp2 + fn2)
    # f12 = 2 * precision2 * recall2 / (precision2 + recall2)
    # bal_acc2 = balanced_accuracy_score(test_labels, predictions2)

    with open(model.prefix + '/metrics.txt', 'w') as f:
        f.write(f"AUC: {auc}\n\n")
        f.write(f"Best val balanced acc threshold:\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"bal_acc: {bal_acc}\n")
        f.write(f"TP: {tp}\n")
        f.write(f"TN: {tn}\n")
        f.write(f"FP: {fp}\n")
        f.write(f"FN: {fn}\n\n")
        # f.write(f"Max normal val loss threshold:\n")
        # f.write(f"Precision: {precision2}\n")
        # f.write(f"Recall: {recall2}\n")
        # f.write(f"F1: {f12}\n")
        # f.write(f"bal_acc: {bal_acc2}\n")
        # f.write(f"TP: {tp2}\n")
        # f.write(f"TN: {tn2}\n")
        # f.write(f"FP: {fp2}\n")
        # f.write(f"FN: {fn2}\n\n")

    with open(model.prefix + '/scores.pkl', 'wb') as f:
        pickle.dump(test_scores, f)
    with open(model.prefix + '/labels.pkl', 'wb') as f:
        pickle.dump(test_labels, f)

    model.log("metrics/test/prec", float(precision))
    model.log("metrics/test/recall", float(recall))
    model.log("metrics/test/f1", float(f1))
    model.log("metrics/test/bal_acc", float(bal_acc))
    model.log("metrics/test/auc", float(auc))

    # model.log("metrics/test/prec2", float(precision2))
    # model.log("metrics/test/recall2", float(recall2))
    # model.log("metrics/test/f12", float(f12))
    # model.log("metrics/test/bal_acc2", float(bal_acc2))

    if not model.args.no_maha_threshold:
        test_maha_scores = np.hstack(model.test_maha_scores)
        maha_predictions = np.where(test_maha_scores > model.maha_thresh, True, False)
        auc_maha = roc_auc_score(test_labels, test_maha_scores)
        tn_maha, fp_maha, fn_maha, tp_maha = confusion_matrix(test_labels, maha_predictions).ravel()
        precision_maha = tp_maha / (tp_maha + fp_maha)
        recall_maha = tp_maha / (tp_maha + fn_maha)
        f1_maha = 2 * precision_maha * recall_maha / (precision_maha + recall_maha)
        bal_acc_maha = balanced_accuracy_score(test_labels, maha_predictions)

        with open(model.prefix + '/metrics.txt', 'a') as f:
            f.write(f"Mahalanobis threshold:\n")
            f.write(f"AUC: {auc_maha}\n")
            f.write(f"Precision: {precision_maha}\n")
            f.write(f"Recall: {recall_maha}\n")
            f.write(f"F1: {f1_maha}\n")
            f.write(f"bal_acc: {bal_acc_maha}\n")
            f.write(f"TP: {tp_maha}\n")
            f.write(f"TN: {tn_maha}\n")
            f.write(f"FP: {fp_maha}\n")
            f.write(f"FN: {fn_maha}\n\n")

        model.log("metrics/test/prec_maha", float(precision_maha))
        model.log("metrics/test/recall_maha", float(recall_maha))
        model.log("metrics/test/f1_maha", float(f1_maha))
        model.log("metrics/test/bal_acc_maha", float(bal_acc_maha))
        model.log("metrics/test/auc_maha", float(auc_maha))