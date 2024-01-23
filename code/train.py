import pickle as pickle
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, get_cosine_schedule_with_warmup
from load_data_copy import *

from torch.optim import NAdam
from transformers import EarlyStoppingCallback


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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, balancing=True):
        super(FocalLoss, self).__init__()
        self.balancing_alpha = { # key means label that changed to number
             0: 0.75, #
             1: 0.75, #
             6: 0.75, #
             4: 0.75, #
            20: 0.75, #
             5: 0.75, #
            15: 0.75, #
             7: 0.75, #
            25: 0.75, #
            12: 0.75, #
            17: 0.75, #
            14: 0.25, #
            21: 0.25, #
            18: 0.25, #
             2: 0.25, #
            24: 0.25, #
             3: 0.25, #
            10: 0.25, #
            11: 0.25, #
            13: 0.25, #
            26: 0.25, #
            28: 0.25, #
             8: 0.25, #
            16: 0.25, #
            19: 0.25, #
            29: 0.25, #
            23: 0.25, #
            22: 0.25, #
             9: 0.25, #
            27: 0.25 # 
            }
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.balancing = balancing

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)

        if self.balancing : 
            targets_copy = targets.tolist()
            # print("targets", targets_copy)
            alpha_t = torch.Tensor([self.balancing_alpha[k] for k in targets_copy]).view(-1, 1).to('cuda')
            F_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    # forward pass
    outputs = model(**inputs)
    logits = outputs.get("logits")  
    # compute custom loss (suppose one has 3 labels with different weights)
    # loss_fct = FocalLoss()
    loss_fct = LabelSmoothingLoss()
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")  
        # compute custom loss (suppose one has 3 labels with different weights)
        #loss_fct = FocalLoss()
        loss_fct = LabelSmoothingLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
def train():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    # MODEL_NAME = "klue/bert-base"
    MODEL_NAME = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=['#', '@'])
    
    # load dataset
    train_dataset = load_data("../dataset/train/train_ffinal.csv")
    dev_dataset = load_data("../dataset/train/dev_ffinal.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # subj_start_id = tokenizer.convert_tokens_to_ids(["<S:PER>", "<S:ORG>"])
    # subj_end_id = tokenizer.convert_tokens_to_ids(["</S:PER>", "</S:ORG>"])
    # obj_start_id = tokenizer.convert_tokens_to_ids(["<O:PER>", "<O:ORG>", "<O:LOC>", "<O:DAT>", "<O:POH>", "<O:NOH>"])
    # obj_end_id = tokenizer.convert_tokens_to_ids(["</O:PER>", "</O:ORG>", "</O:LOC>", "</O:DAT>", "</O:POH>", "</O:NOH>"])
    # start_id = subj_start_id+obj_start_id
    # end_id   = subj_end_id+obj_end_id

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    train_ent_pos_emb = get_entity_position_embedding(tokenizer, tokenized_train['input_ids'])
    print(len(tokenized_train['input_ids'][0]))
    for i in train_ent_pos_emb:
        if len(i) == 4:continue
        else:
            print("error")
    #tokenized_train['entity_ids'] = making_entity_pos_emb(train_ent_pos_emb)
    # entity_ids = entity_ids_maker(train_dataset, tokenizer)
    # tokenized_train['entity_ids'] = entity_ids
    
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
    dev_ent_pos_emb = get_entity_position_embedding(tokenizer, tokenized_dev['input_ids'])
    #tokenized_dev['entity_ids'] = making_entity_pos_emb(dev_ent_pos_emb)
    # entity_ids = entity_ids_maker(dev_dataset, tokenizer)
    # tokenized_dev['entity_ids'] = entity_ids

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30
    model_config.classifier_dropout = 0.1
    
    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    
    model.resize_token_embeddings(len(tokenizer))
    print(model.config)
    model.parameters
    model.to(device)
    
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=20,              # total number of training epochs
        learning_rate=4e-5,               # learning_rate
        per_device_train_batch_size=48,  # batch size per device during training
        per_device_eval_batch_size=48,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step. (원래 steps였음)
        save_strategy='epoch',  # 이 줄 변경
        fp16=True,                                    
        load_best_model_at_end = True 
    )

    early_stopping = EarlyStoppingCallback(
      early_stopping_patience=4,  # Patience 값 설정 (일정 에폭동안 검증 손실이 개선되지 않으면 중단)
      early_stopping_threshold=0.02,  # 검증 손실의 개선이 얼마나 작아야 하는지 설정
    )
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)  # Adjust the learning rate as needed
    optimizer = torch.optim.NAdam(model.parameters(), lr=training_args.learning_rate)  # Adjust the learning rate as needed
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        callbacks=[early_stopping],
        optimizers=(optimizer, scheduler)
    )
    # trainer.optimizer = torch.optim.AdamW
    # trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    
    # train model
    trainer.train()
    model.save_pretrained('./best_model')

def main():
  train()

if __name__ == '__main__':
  main()
