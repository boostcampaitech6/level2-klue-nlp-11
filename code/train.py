import pickle as pickle
import pandas as pd
import torch
import numpy as np
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
import wandb
import yaml
import os
import random
from pytorch_lightning import seed_everything


from trainer import *
from model import *
from util.util import *
from load_data import *

with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config.yaml') as f:
    config = yaml.safe_load(f)

seed = config['SEED']
seed_everything(seed)


def train():
    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    MODEL_NAME = config['MODEL_NAME']

    pretrained_model_path = config['PRETRAINED_MODEL_PATH']

    model_config = AutoConfig.from_pretrained(pretrained_model_path)
    model_config.num_labels = 30  # Set the number of labels if it's a classification task

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, config=model_config)

    tokenizer.add_tokens(['§'])

    dataset = load_data(config['TRAIN_PATH'], config['MODEL_TYPE'], config['DISCRIP'], config['DO_SEQUENTIALBERTMODEL'])
    train_dataset, dev_dataset = train_dev_split(dataset, config['RATIO'])

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    tokenized_train =  tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    model = SpecialPunctBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer)

    model.to(device)

    wandb.init(project=config['PROJECT_NAME'], name=config['RUN_NAME'])
    training_args = TrainingArguments(
        output_dir=config['OUTPUT_DIR'],
        save_total_limit=config['TOTAL_SAVE_MODEL'],
        save_steps=config['SAVING_STEP'],
        num_train_epochs=config['MAX_EPOCH'],
        learning_rate=config['LR'],
        per_device_train_batch_size=config['BATCH_SIZE'],
        per_device_eval_batch_size=config['BATCH_SIZE'],
        warmup_steps=config['WARMUP_STEP'],
        weight_decay=config['WEIGHT_DECAY'],
        logging_dir=config['LOGGING_DIR'],
        logging_steps=config['LOGGING_STEP'],
        logging_strategy=config['STRATEGY'],
        save_strategy=config['STRATEGY'],
        evaluation_strategy=config['STRATEGY'],
        eval_steps=config['EVAL_STEP'],
        load_best_model_at_end=True,
        report_to="wandb",
        metric_for_best_model='micro f1 score')

    trainer = CustomTrainer(
        loss_fn=config['LOSS_FN'],
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
        eval_dataset=RE_dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    trainer.train()
    model.save_pretrained(config['MODEL_SAVE_DIR'])
    wandb.finish()



def kfoldtrain():

    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(config['MODEL_SAVE_DIR'], exist_ok=True)

    MODEL_NAME = config['MODEL_NAME']

    pretrained_model_path = config['PRETRAINED_MODEL_PATH']

    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
    tokenizer.add_tokens(['§'])
    model_config = AutoConfig.from_pretrained(config['PRETRAINED_MODEL_PATH'])
    model_config.num_labels = 30 

    dataset = load_data(config['TRAIN_PATH'], config['MODEL_TYPE'], discrip=1)
    labels = label_to_num(dataset['label'].values)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    data_collator = DataCollatorWithPadding(tokenizer)

    fold_results = []

    for fold, (train_idx, dev_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"FOLD {fold}")
        print("TRAIN:", train_idx, "TEST:", dev_idx)
        
        wandb.init(project=config['PROJECT_NAME'], name=f"{config['RUN_NAME']}_fold{fold}", reinit=True)
        
        train_dataset = dataset.iloc[train_idx]
        dev_dataset = dataset.iloc[dev_idx]

        train_label = label_to_num(train_dataset['label'].values)
        dev_label = label_to_num(dev_dataset['label'].values)
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

        model = SpecialPunctBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)
        model.to(device)

        training_args = TrainingArguments(
            output_dir=config['OUTPUT_DIR'],
            save_total_limit=config['SAVE_MODEL_NUM'],
            save_steps=config['SAVING_STEP'],
            num_train_epochs=config['MAX_EPOCH'],
            learning_rate=config['LR'],
            per_device_train_batch_size=config['BATCH_SIZE'],
            per_device_eval_batch_size=config['BATCH_SIZE'],
            warmup_steps=config['WARMUP_STEP'],
            weight_decay=config['WEIGHT_DECAY'],
            logging_dir=config['LOGGING_DIR'],
            logging_steps=config['LOGGING_STEP'],
            logging_strategy=config['STRATEGY'],
            save_strategy=config['STRATEGY'],
            evaluation_strategy=config['STRATEGY'],
            eval_steps=config['EVAL_STEP'],
            load_best_model_at_end=True,
            report_to="wandb",
            metric_for_best_model='micro f1 score')
        
        trainer = CustomTrainer(
            loss_fn=config['LOSS_FN'],
            model=model,
            args=training_args,
            train_dataset=RE_train_dataset,
            eval_dataset=RE_dev_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator)
        
        try:
            trainer.train()
            fold_results.append(trainer.evaluate())
            model.save_pretrained(f"{config['MODEL_SAVE_DIR']}_fold{fold}")
        except Exception as e:
            print(f"Error occurred in fold {fold}: {e}")

        wandb.finish()

    avg_results = {metric: np.mean([result[metric] for result in fold_results]) for metric in fold_results[0]}
    print("Average results across folds:", avg_results)

if __name__ == '__main__':
    seed_everything()
  
    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config.yaml') as f:
        config = yaml.safe_load(f)
    
    kfoldtrain()