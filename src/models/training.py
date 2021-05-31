import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from src.models.metrics import calculate_metrics
from src.models.data_loader import create_data_loader
from src.models.models import HIBERT

from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup

class Trainer():

    def __init__(self, model_dir, model_config,
                max_len=512*4, epochs=4, out_dir=None, device='cpu',
                loss_fn='cross_entropy_loss', optimizer='adamw', lr=1e-5, scheduler='linear_schedule_with_warmup',
                train_batch_size=32, val_batch_size=32, test_batch_size=32, chunksize=512, sampler = None):
        
        self.out_dir = out_dir
        self.model_dir = model_dir
        self.model_config = model_config
        self.epochs = epochs
        self.device = device

        # load model and tokenizer
        self.model = HIBERT.from_pretrained(self.model_dir, **self.model_config)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)

        # loss function, optimizer and scheduler
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler

        # dataloader parameters
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.chunksize = chunksize
        self.sampler = sampler
        self.shuffle = None

        if self.sampler is None:
            self.shuffle = True
        else:
            self.shuffle = False

        # other attributes
        self.history = None
        self.best_threshold = None
        self.best_f1 = None

    # loss function
    def get_loss_function(self, loss_fn, device):

        if loss_fn == 'cross_entropy_loss':
            out = nn.CrossEntropyLoss().to(device)
        
        return out


    def get_scheduler(self, scheduler, optimizer, total_steps):
        
        if scheduler == 'linear_schedule_with_warmup':
            out = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
                )
        
        return out


    def get_optimizer(self, model, optimizer, lr):

        if optimizer == 'adamw':
            out = AdamW(model.parameters(), lr=lr, correct_bias=False)
            
        return out


    # training for one epoch
    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler):

        # Set to training mode
        model = model.train()

        losses = []
        pred_prob_list = []
        target_list = []

        # training loop for one epoch
        for d in data_loader:
            input_ids = d['input_ids'].type(torch.LongTensor).to(device)
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
            loss.backward()

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
            calculate_metrics(
                y_true=np.concatenate(target_list), 
                probas_pred=np.concatenate(pred_prob_list)
            )

        loss_epoch = np.mean(losses)

        return loss_epoch, acc_score_epoch, best_f1_epoch, auc_score_epoch, best_threshold_epoch


    # evaluate model on validation set (based on best threshold)
    def eval_model(self, model, data_loader, loss_fn, device, best_threshold):

        # Turn on evaluation mode which disables dropout.
        model = model.eval()

        losses = []
        pred_prob_list = []
        target_list = []

        # evaluate model through the validation set
        # torch.no_grad() for speed and also since backprop is not needed
        with torch.no_grad():

            for d in data_loader:
                input_ids = d['input_ids'].type(torch.LongTensor).to(device)
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
            calculate_metrics(
                y_true=np.concatenate(target_list), 
                probas_pred=np.concatenate(pred_prob_list), 
                best_threshold=best_threshold
                )

        loss_epoch = np.mean(losses)

        return loss_epoch, acc_score_epoch, best_f1_epoch, auc_score_epoch


    # main training loop
    def train_model(self, epochs, model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler):

        self.history = defaultdict(list)
        best_f1 = 0

        # training loop
        for epoch in range(epochs):

            print(f'Epoch {epoch + 1} / {epochs}')
            print('-'*20)

            train_loss, train_acc, train_f1, train_auc, best_threshold = self.train_epoch(
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

            val_loss, val_acc, val_f1, val_auc = self.eval_model(
                model = model, 
                data_loader = val_data_loader, 
                loss_fn = loss_fn, 
                device = device,
                best_threshold = best_threshold
            )

            print(
                f'Epoch val   loss: {round(val_loss, 4):.4f}     ' + 
                f'ACC: {round(val_acc, 4):.4f}     ' + 
                f'F1: {round(val_f1, 4):.4f}     ' + 
                f'AUC: {round(val_auc, 4):.4f}     '
            )

            print()

            # save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['train_auc'].append(train_auc)

            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)

            self.history['best_threshold'].append(best_threshold)

            # save best model
            if val_f1 > best_f1:

                # save model
                model.save_pretrained(Path(self.out_dir, 'trained_model'))
                self.tokenizer.save_pretrained(Path(self.out_dir, 'tokenizer'))
                
                # save threshold
                pickle.dump(best_threshold, open(Path(self.out_dir, 'best_threshold.pkl'), 'wb'))
                self.best_threshold = best_threshold
                
                # update f1
                best_f1 = val_f1
                self.best_f1 = best_f1


    # Prediction function
    def pred_model(self, model, data_loader, device, best_threshold):

        # Turn on evaluation mode which disables dropout.
        model = model.eval()

        pred_prob_list = []
        target_list = []
        attn_wts_list = []
        input_ids_list = []
        n_chunks_list = []

        # make predictions on test set
        # torch.no_grad() for speed and also since backprop is not needed
        with torch.no_grad():
            
            for d in data_loader:
                input_ids = d['input_ids'].type(torch.LongTensor).to(device)
                attention_mask = d['attention_mask'].to(device)
                n_chunks = d['n_chunks'].to(device)
                target = d['target'].to(device)

                # pass through model
                outputs = model(
                    input_ids = input_ids, 
                    attention_mask = attention_mask, 
                    n_chunks = n_chunks
                )

                # make predictions
                pred_prob = F.softmax(outputs, 1)[:, 1]

                # collect targets and predicted prob for epoch level metrics
                target_list.append(target)
                pred_prob_list.append(pred_prob)
                attn_wts_list.extend(model.attn_weights)
                input_ids_list.append(input_ids)
                n_chunks_list.extend(n_chunks.tolist())

        # concat output outside the loop
        target_array = torch.cat(target_list).tolist()
        pred_prob_array = torch.cat(pred_prob_list).tolist()
        pred_array = (pd.Series(pred_prob_array) > best_threshold).astype(int).tolist()

        # compile attention weights
        attn_wts_compiled = [i.numpy().flatten() for i in attn_wts_list]

        # compile input_ids
        input_ids_array = torch.cat(input_ids_list)
        input_ids_chunks = input_ids_array.split_with_sizes(n_chunks_list)

        input_ids_compiled = []
        for chunk in input_ids_chunks:
            input_ids_compiled.append(chunk.view(1, -1).numpy().flatten())

        output = {
            'target': target_array,
            'pred_proba': pred_prob_array,
            'pred': pred_array,
            'attn_wts': attn_wts_compiled,
            'input_ids': input_ids_compiled,
        }

        return output


    # call this function for training
    def fit(self, df_train, df_val):

        # create data loaders
        train_data_loader = create_data_loader(
            df = df_train, 
            tokenizer = self.tokenizer, 
            max_len = self.max_len, 
            batch_size = self.train_batch_size, 
            chunksize = self.chunksize, 
            sampler = self.sampler, 
            shuffle = self.shuffle, 
            drop_last = True)

        val_data_loader = create_data_loader(
            df = df_val, 
            tokenizer = self.tokenizer, 
            max_len = self.max_len, 
            batch_size = self.val_batch_size, 
            chunksize = self.chunksize, 
            sampler = None, 
            shuffle = False, 
            drop_last = False)

        # get loss function, optimizer and scheduler
        loss_fn = self.get_loss_function(loss_fn=self.loss_fn, device=self.device)
        optimizer = self.get_optimizer(model=self.model, optimizer=self.optimizer, lr=self.lr)
        scheduler = self.get_scheduler(scheduler=self.scheduler, optimizer=optimizer, 
                                        total_steps=(len(train_data_loader) * self.epochs))

        # call main training loop
        self.train_model(
            epochs=self.epochs, 
            model=self.model, 
            train_data_loader=train_data_loader, 
            val_data_loader=val_data_loader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=self.device, 
            scheduler=scheduler)


    # call this function for prediction
    def predict(self, df_test):
        
        # create test data loader
        test_data_loader = create_data_loader(
            df = df_test, 
            tokenizer = self.tokenizer, 
            max_len = self.max_len, 
            batch_size = self.batch_size, 
            chunksize = 512, 
            sampler = None, 
            shuffle = False, 
            drop_last = False)
        
        # get best threshold
        if self.best_threshold is None:
            best_threshold = 0.5
        else:
            best_threshold = self.best_threshold

        # make predictions
        return pred_model(
            model=self.model, 
            data_loader=test_data_loader, 
            device=self.device, 
            best_threshold=best_threshold
            )