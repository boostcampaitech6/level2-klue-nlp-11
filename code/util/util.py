import wandb
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle
import random
import os
import torch
import torch.nn as nn




label_list = ['no_relation', 'org:top_members/employees', 'org:members', 'org:product', 'per:title', 'org:alternate_names',
                'per:employee_of', 'org:place_of_headquarters', 'per:product',
                'org:number_of_employees/members', 'per:children',
                'per:place_of_residence', 'per:alternate_names',
                'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                'org:member_of', 'per:parents', 'org:dissolved',
                'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                'per:religion']
  
def viz(labels, preds, probs):
    wandb.log({
            "auprc": wandb.plot.roc_curve(labels, probs, labels=label_list),
            "precision_recall": wandb.plot.pr_curve(labels, probs, labels=label_list),  
            "Confusion Matrix": wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=label_list)
        })

def klue_re_micro_f1(preds, labels):
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)
    
    viz(labels, preds, probs)
    
    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc
    }

def label_to_num(label):
    num_label = []
    with open('./dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])
  
    return num_label

def num_to_label(label):
    origin_label = []

    with open('./dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    
    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label


def add_token(tokenizer, model_type):
    if model_type == 'entity_special':
        tokenizer.add_special_tokens({"additional_special_tokens":['[S:PER]', '[/S:PER]', '[O:PER]', '[/O:PER]',
            '[S:ORG]', '[/S:ORG]', '[O:ORG]', '[/O:ORG]', '[O:POH]', '[/O:POH]',
            '[O:DAT]', '[/O:DAT]', '[S:LOC]', '[/S:LOC]', '[O:LOC]', '[/O:LOC]', '[O:NOH]', '[/O:NOH]']})
        
    elif model_type == 'entity_punct' or model_type == 'new_entity_punct' or model_type == 'ko_entity_punct':
        tokenizer.add_tokens(['§'])

    elif model_type == 'cls_entity_special':
        new_special_tokens = {"additional_special_tokens" : ['[SUBJ]' , '[OBJ]' , '[PER]' , '[ORG]',
            '[DAT]' , '[LOC]' , '[POH]' , '[NOH]']}
        tokenizer.add_special_tokens(new_special_tokens)
    
    return tokenizer



def seed_everything(seed: int=14):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=30, smoothing=0.2, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def train_dev_split(dataset, dev_size: float, random_state: int = 42):

    dev_size = max(dev_size, 0.1)

    label_counts = defaultdict(int)

    for label in dataset['label']:
        label_counts[label] += 1

    train_data_list = []
    val_data_list = []

    for label, count in tqdm(label_counts.items(), desc='train_validation_split', total=len(label_counts)):
        label_data = dataset[dataset['label'] == label]
        train_data, val_data = train_test_split(label_data, test_size=dev_size, random_state=random_state)
        train_data_list.append(train_data)
        val_data_list.append(val_data)

    train_dataset = pd.concat(train_data_list, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_dataset = pd.concat(val_data_list, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_dataset, val_dataset