import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from src.models.metrics import calculate_metrics

# training for one epoch
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):

    # Set to training mode
    model = model.train()

    losses = []
    pred_prob_list = []
    target_list = []

    # training loop for one epoch
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        n_chunks = d['n_chunks'].to(device)
        target = d['target'].to(device)

        # pass through model
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            n_chunks = n_chunks
        )

        # make predictions and calculate loss
        pred_prob = F.softmax(outputs, 1)[:, 1]
        loss = loss_fn(outputs, target)

        # collect loss
        losses.append(loss.item())

        # compulsory steps
        optimizer.zero_grad()
        loss.backwards()

        # clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # detach tensors for metrics
        target_np = target.detach().numpy()
        pred_prob_np = pred_prob.detach().numpy()

        # collect targets and predicted prob for epoch level metrics
        target_list.append(target_np)
        pred_prob_list.append(pred_prob_np)

        # batch level metrics
        acc_score, best_f1, auc_score, _ = calculate_metrics(y_true=target_np, probas_pred=pred_prob_np)

        # print batch level metrics
        print(
            f'Batch Train loss: {round(loss.item(), 4):.4f}     ' + 
            f'Class 1 prop: {round(np.mean(target_np), 4):.4f}     ' + 
            f'ACC: {round(acc_score, 4):.4f}     ' + 
            f'F1: {round(best_f1, 4):.4f}     ' + 
            f'AUC: {round(auc_score, 4):.4f}     '
        )

    # Epoch level metrics (also return best threshold)
    acc_score_epoch, best_f1_epoch, auc_score_epoch, best_threshold_epoch = \
        calculate_metrics(y_true=target_list, probas_pred=pred_prob_list)

    loss_epoch = np.mean(losses)

    return loss_epoch, acc_score_epoch, best_f1_epoch, auc_score_epoch, best_threshold_epoch


# evaluate model on validation set (based on best threshold)
def eval_model(model, data_loader, loss_fn, device, best_threshold):

    # Set to training mode
    model = model.eval()

    losses = []
    pred_prob_list = []
    target_list = []

    # evaluate model through the validation set
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        n_chunks = d['n_chunks'].to(device)
        target = d['target'].to(device)

        # pass through model
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            n_chunks = n_chunks
        )

        # make predictions and calculate loss
        pred_prob = F.softmax(outputs, 1)[:, 1]
        loss = loss_fn(outputs, target)

        # collect loss
        losses.append(loss.item())

        # detach tensors for metrics
        target_np = target.detach().numpy()
        pred_prob_np = pred_prob.detach().numpy()

        # collect targets and predicted prob for epoch level metrics
        target_list.append(target_np)
        pred_prob_list.append(pred_prob_np)

    # Epoch level metrics (use threshold from training to make predictions)
    acc_score_epoch, best_f1_epoch, auc_score_epoch, _ = \
        calculate_metrics(y_true=target_list, probas_pred=pred_prob_list, best_threshold=best_threshold)

    loss_epoch = np.mean(losses)

    return loss_epoch, acc_score_epoch, best_f1_epoch, auc_score_epoch


# main training loop
def train_model(epochs, model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler):

    history = defaultdict(list)
    best_f1 = 0

    # training loop
    for epoch in range(epochs):

        print(f'Epoch {epoch + 1} / {epochs}')
        print('-'*20)

        train_loss, train_acc, train_f1, train_auc, best_threshold = train_epoch(
            model = model, 
            data_loader = train_data_loader, 
            loss_fn = loss_fn, 
            optimizer = optimizer, 
            device = device, 
            scheduler = scheduler
        )

        print()
        print(
            f'Epoch Train loss: {round(train_loss, 4):.4f}     ' + 
            f'ACC: {round(train_acc, 4):.4f}     ' + 
            f'F1: {round(train_f1, 4):.4f}     ' + 
            f'AUC: {round(train_auc, 4):.4f}     '
        )

        val_loss, val_acc, val_f1, val_auc = eval_model(
            model = model, 
            data_loader = val_data_loader, 
            loss_fn = loss_fn, 
            device = device,
            best_threshold = best_threshold
        )

        print(
            f'Epoch val  loss: {round(val_loss, 4):.4f}     ' + 
            f'ACC: {round(val_acc, 4):.4f}     ' + 
            f'F1: {round(val_f1, 4):.4f}     ' + 
            f'AUC: {round(val_auc, 4):.4f}     '
        )

        print()

        # save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        history['best_threshold'].append(best_threshold)

        # save best model
        if val_f1 > best_f1:

            # save model
            torch.save(model.state_dict(), 'best_model_state.bin')
            
            # save threshold
            pickle.dump(best_threshold, open('best_threshold.pkl', 'wb'))
            
            # update f1
            best_f1 = val_f1