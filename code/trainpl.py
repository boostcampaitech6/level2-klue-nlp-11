import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from typing import List
from torch.utils.data import DataLoader
import transformers
import sklearn
import pickle as pickle

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer


import pandas as pd
import os
import numpy as np
import random

import yaml
config = yaml.load(open('./config.yaml', 'r'), Loader = yaml.Loader)
sweep = yaml.load(open('./sweep.yaml', 'r'), Loader = yaml.Loader)

from metrics import *

wandb_logger = WandbLogger(
    project=config['project_name'], 
    entity='xi-vil',
    config=sweep,
    name=config['run_name'])##config에서 runname 설정필요!!

class Model(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 lr, 
                 weight_decay,
                 loss_fn,
                 warmup_steps,
                 total_steps,
                 lr_scheduler_type, 
                 ):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_scheduler_type = lr_scheduler_type

        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.num_labels = 30

        # model 선택
        self.plm = AutoModelForSequenceClassification.from_pretrained(model_name, config = self.model_config)

        # loss function 선택
        ####개발필요


        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def set_seed(seed:int = config['seed']):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def compute_metrics(pred):
        """ validation을 위한 metrics function """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = pred.predictions

        # calculate accuracy using sklearn's function
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

        return {
            'micro f1 score': f1,
            'auprc' : auprc,
            'accuracy': acc,
            }

    def label_to_num(label):
        num_label = []
        with open('dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)
        for v in label:
            num_label.append(dict_label_to_num[v])
        
        return num_label

    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):
        output = self.plm(input_ids, attention_mask, token_type_ids, labels = labels)
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        output = self(input_ids, attention_mask, token_type_ids, labels)
        loss = output.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)

        f1 = klue_re_micro_f1(output.logits.argmax(-1).cpu().numpy(), labels.cpu().numpy())
        self.log('train_f1', f1, prog_bar=True, logger=True)
        auprc = klue_re_auprc(output.logits.cpu().numpy(), labels.cpu().numpy())
        self.log('train_auprc', auprc, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        output = self(input_ids, attention_mask, token_type_ids, labels)
        loss = output.loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

        f1 = klue_re_micro_f1(output.logits.argmax(-1).cpu().numpy(), labels.cpu().numpy())
        self.log('val_f1', f1, prog_bar=True, logger=True)
        auprc = klue_re_auprc(output.logits.cpu().numpy(), labels.cpu().numpy())
        self.log('val_auprc', auprc, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x = {"input_ids": batch["input_ids"], "token_type_ids": batch["token_type_ids"], "attention_mask": batch["attention_mask"]}
        y = batch["labels"]
        logits = self(x)

    
    def predict_step(self, batch, batch_idx):
        x = {"input_ids": batch["input_ids"], "token_type_ids": batch["token_type_ids"], "attention_mask": batch["attention_mask"]}
        y = batch["labels"]
        logits = self(x)
        return logits.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
            )
        if self.lr_scheduler_type == 'linear':
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
                )
        elif self.lr_scheduler_type == 'cosine':
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
                )
        elif self.lr_scheduler_type == 'inverse_sqrt':
            scheduler = transformers.get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=self.warmup_steps
                )
        
        return [optimizer], [scheduler]
    
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, save_last = True, **kwargs):
        super().__init__(**kwargs)
        self.save_last = save_last

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if current is None:
            return

        self._save_model(trainer, pl_module, metrics)

        if self.save_last:
            last_filepath = self._format_checkpoint_name(trainer, pl_module, metrics, "last")
            trainer.save_checkpoint(last_filepath)


