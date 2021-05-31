from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score, accuracy_score
import pandas as pd
import numpy as np


def calculate_best_threshold(y_true, probas_pred):

    precision, recall, threshold = \
        precision_recall_curve(y_true, probas_pred)

    f1_score_ = 2*((precision*recall)/(precision+recall))
    metric_df = pd.DataFrame({'precision':precision[1:], 'recall':recall[1:], 'f1_score':f1_score_[1:], 'threshold':threshold})
    metric_df = metric_df.sort_values('f1_score', ascending = False)
        
    # get threshold based on best F1
    best_threshold = metric_df['threshold'].values[0]

    return best_threshold


def calculate_metrics(y_true, probas_pred, best_threshold=0.5):
    
    precision, recall, _ = \
        precision_recall_curve(y_true, probas_pred)

    # Calculate auc
    auc_score = auc(recall, precision)

    # make predictions based on best threshold
    y_pred = (pd.Series(probas_pred) > best_threshold).astype(int)

    # Calculate F1 and acc based on best threshold
    best_f1 = f1_score(y_true=y_true, y_pred=y_pred)
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

    out = {
        'Class 1 prop': np.mean(y_true),
        'ACC': acc_score,
        'F1': best_f1,
        'AUC': auc_score
        }

    return out

