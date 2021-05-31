from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score, accuracy_score
import pandas as pd

def calculate_metrics(y_true, probas_pred, best_threshold=None):
    
    precision, recall, threshold = \
        precision_recall_curve(y_true, probas_pred)

    # Calculate auc
    auc_score = auc(recall, precision)

    if best_threshold is None:
        # calculate precision, recall, f1 with varying thresholds
        f1_score_ = 2*((precision*recall)/(precision+recall))
        metric_df = pd.DataFrame({'precision':precision[1:], 'recall':recall[1:], 'f1_score':f1_score_[1:], 'threshold':threshold})
        metric_df = metric_df.sort_values('f1_score', ascending = False)
        
        # get threshold based on best F1
        best_threshold = metric_df['threshold'].values[0]

    # make predictions based on best F1
    y_pred = (pd.Series(probas_pred) > best_threshold).astype(int)

    # Calculate F1 and acc based on best threshold
    best_f1 = f1_score(y_true=y_true, y_pred=y_pred)
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

    out = {
        'acc_score': acc_score,
        'best_f1': best_f1,
        'auc_score': auc_score, 
        'best_threshold':best_threshold
        }

    return out

