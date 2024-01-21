import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from transformers import AutoModelForMaskedLM, AutoModel
from load_data import *
from sklearn.model_selection import StratifiedKFold
# from traindevsplit import * # train_dev_split
import numpy as np
import random

import yaml
config = yaml.load(open('./config.yaml', 'r'), Loader = yaml.Loader)
sweep = yaml.load(open('./sweep.yaml', 'r'), Loader = yaml.Loader)

import wandb
wandb.init(project=config['project_name'], entity='xi-vil')
run = wandb.init(config=sweep)


MARKERS = dict(
    subject_start_marker="<SUB>",
    subject_end_marker  ="</SUB>",
    object_start_marker ="<OBJ>",
    object_end_marker   ="</OBJ>",
)
TYPE_MARKERS = dict(
    subject_start_per_marker="<S:PER>",
    subject_start_org_marker="<S:ORG>",
    subject_start_loc_marker="<S:LOC>",
    subject_end_per_marker ="</S:PER>",
    subject_end_org_marker ="</S:ORG>",
    subject_end_loc_marker="</S:LOC>",
    object_start_per_marker="<O:PER>",
    object_start_org_marker="<O:ORG>",
    object_start_loc_marker="<O:LOC>",
    object_start_dat_marker="<O:DAT>",
    object_start_poh_marker="<O:POH>",
    object_start_noh_marker="<O:NOH>",
    object_end_per_marker ="</O:PER>",
    object_end_org_marker ="</O:ORG>",
    object_end_loc_marker ="</O:LOC>",
    object_end_dat_marker ="</O:DAT>",
    object_end_poh_marker ="</O:POH>",
    object_end_noh_marker ="</O:NOH>",
)



def set_seed(seed:int = config['seed']):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

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

def train():
  set_seed(seed=config['seed'])
  # load model and tokenizer
  MODEL_NAME = config['MODEL_NAME']
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  new_special_tokens = list(TYPE_MARKERS.values())
  punt_special_tokens = ["*","@","#"]

  tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

  #   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)

  # load dataset
  dataset = load_data(config['train']['train_dataset_filepath'])
  labels = label_to_num(dataset['label'].values)

  # K-Fold Cross-Validation
  n_splits = 5  # Number of folds
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['seed'])

  fold_results = []



  for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
      print(f"Training fold {fold + 1}/{n_splits}")

      # Splitting dataset into train and validation for the current fold
      train_dataset = dataset.iloc[train_idx]
      val_dataset = dataset.iloc[val_idx]

      train_label = [labels[i] for i in train_idx]
      val_label = [labels[i] for i in val_idx]

      # Tokenizing dataset
      tokenized_train = tokenized_dataset(train_dataset, tokenizer)
      tokenized_val = tokenized_dataset(val_dataset, tokenizer)

      # Creating dataset for PyTorch
      RE_train_dataset = RE_Dataset(tokenized_train, train_label)
      RE_val_dataset = RE_Dataset(tokenized_val, val_label)

    #   # Setting model hyperparameter
    #   model_path = './pretrained_roberta_large'
    #   model_config = AutoConfig.from_pretrained(f'{model_path}/config.json')
    #   model_config.num_labels = 30

    #   # Load the TAPT model with its configuration
    #   tapt_model = AutoModelForMaskedLM.from_pretrained(model_path, config=model_config)
    #   bert_state_dict = {k: v for k, v in tapt_model.state_dict().items() if k.startswith("roberta")}
      
      model_config =  AutoConfig.from_pretrained("klue/roberta-large")
      model_config.num_labels = 30
      # Load the sequence classification model
      seq_model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", config=model_config)
      seq_model.resize_token_embeddings(len(tokenizer))

    #   missing_keys, unexpected_keys = seq_model.load_state_dict(bert_state_dict, strict=False)
      

      # Print the config and parameter details of the sequence classification model
      print(seq_model.config)
      print(seq_model.parameters)

      # Ensure 'device' is defined (e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
      seq_model.to(device)

      #Training setting
      training_args = TrainingArguments(
        output_dir=config['train']['training_args']['output_dir'],                # output directory
        run_name=config['train']['training_args']['run_name'],
        report_to=config['train']['training_args']['report_to'],
        save_total_limit=config['train']['training_args']['save_total_limit'],    # number of total save model.
        save_steps=config['train']['training_args']['save_steps'],                # model saving step.
        logging_dir=config['train']['training_args']['logging_dir'],                  # directory for storing logs
        logging_steps=config['train']['training_args']['logging_steps'],              # log saving step.
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
                                                                                      # `no`: No evaluation during training.
                                                                                      # `steps`: Evaluate every `eval_steps`.
                                                                                      # `epoch`: Evaluate every end of epoch.
        eval_steps = config['train']['training_args']['eval_steps'],                  # evaluation step.
        save_strategy='steps',       # checkpoint saving strategy to adopt during training
        load_best_model_at_end = config['train']['training_args']['load_best_model_at_end'],

        learning_rate=config['train']['training_args']['learning_rate'],       # learning_rate
        num_train_epochs=config['train']['training_args']['num_train_epochs'],    # total number of training epochs
        per_device_train_batch_size=config['train']['training_args']['per_device_train_batch_size'],  # batch size per device during training
        per_device_eval_batch_size=config['train']['training_args']['per_device_eval_batch_size'],    # batch size for evaluation
        warmup_steps=config['train']['training_args']['warmup_steps'],                # number of warmup steps for learning rate scheduler
        weight_decay=config['train']['training_args']['weight_decay'],                 # strength of weight decay

        fp16=True, # mixed precision training using apex
        # metric_for_best_model= 'micro_f1_score'
        )


      # Train model for the current fold
      trainer = Trainer(
          model=seq_model,
          args=training_args,
          train_dataset=RE_train_dataset,
          eval_dataset=RE_val_dataset,
          compute_metrics=compute_metrics,
          callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
      )

      # Train and save results
      trainer.train()
      fold_results.append(trainer.evaluate())

      # Optionally, save the model for each fold
      seq_model.save_pretrained(f"{config['best_model_dir']}_fold{fold}")

  # Aggregate and display results from all folds
  avg_results = {metric: np.mean([result[metric] for result in fold_results]) for metric in fold_results[0]}
  print("Average results across folds:", avg_results)

def main():
  train()

if __name__ == '__main__':
  main()
