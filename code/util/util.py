import wandb
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle
import random
import os
import torch
import torch.nn as nn
    
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader




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


def add_token(tokenizer):
    tokenizer.add_tokens(['§'])

    return tokenizer



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
    subject_start_dat_marker="<S:DAT>",
    subject_start_poh_marker="<S:POH>",
    subject_start_noh_marker="<S:NOH>",
    
    subject_end_per_marker ="</S:PER>",
    subject_end_org_marker ="</S:ORG>",
    subject_end_loc_marker ="</S:LOC>",
    subject_end_dat_marker ="</S:DAT>",
    subject_end_poh_marker ="</S:POH>",
    subject_end_noh_marker ="</S:NOH>",

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
TYPE_MARKERS_PUNC = dict(
    subject_start_per_marker="# * PER *",
    subject_start_org_marker="# * ORG *",
    subject_start_loc_marker="# * LOC *",
    subject_start_dat_marker="# * DAT *",
    subject_start_poh_marker="# * POH *",
    subject_start_noh_marker="# * NOH *",
    
    # subject_end_per_marker ="</S:PER>",
    # subject_end_org_marker ="</S:ORG>",
    # subject_end_loc_marker ="</S:LOC>",
    # subject_end_dat_marker ="</S:DAT>",
    # subject_end_poh_marker ="</S:POH>",
    # subject_end_noh_marker ="</S:NOH>",

    object_start_per_marker="@ * PER *",
    object_start_org_marker="@ * ORG *",
    object_start_loc_marker="@ * LOC *",
    object_start_dat_marker="@ * DAT *",
    object_start_poh_marker="@ * POH *",
    object_start_noh_marker="@ * NOH *",
    
    # object_end_per_marker ="</O:PER>",
    # object_end_org_marker ="</O:ORG>",
    # object_end_loc_marker ="</O:LOC>",
    # object_end_dat_marker ="</O:DAT>",
    # object_end_poh_marker ="</O:POH>",
    # object_end_noh_marker ="</O:NOH>",
)

def entity_marker(data : pd.Series):
    sent = data['sentence']
    sbj = data['subject_word']
    obj = data['object_word']
    sent = sent.replace(sbj, MARKERS['subject_start_marker']+' '+sbj+' '+MARKERS['subject_end_marker'])
    sent = sent.replace(obj, MARKERS['object_start_marker']+' '+obj+' '+MARKERS['object_end_marker'])
    return sent
# >>>'〈Something〉는 <OBJ> 조지 해리슨 </OBJ>이 쓰고 <SUB> 비틀즈 </SUB>가 1969년 앨범 《Abbey Road》에 담은 노래다.'
def typed_entity_marker(data : pd.Series):
    sent = data['sentence']
    sbj = data['subject_word']
    sbj_start_type_mark = TYPE_MARKERS[f"subject_start_{data['subject_type'].lower()}_marker"]
    sbj_end_type_mark = TYPE_MARKERS[f"subject_end_{data['subject_type'].lower()}_marker"]
    obj = data['object_word']
    obj_start_type_mark = TYPE_MARKERS[f"object_start_{data['object_type'].lower()}_marker"]
    obj_end_type_mark = TYPE_MARKERS[f"object_end_{data['object_type'].lower()}_marker"]
    sent = sent.replace(sbj, sbj_start_type_mark+' '+sbj+' '+sbj_end_type_mark)
    sent = sent.replace(obj, obj_start_type_mark+' '+obj+' '+obj_end_type_mark)
    return sent
# >>>'〈Something〉는 <O:PER> 조지 해리슨 </O:PER>이 쓰고 <S:ORG> 비틀즈 </S:ORG>가 1969년 앨범 《Abbey Road》에 담은 노래다.'
def typed_entity_marker_punc(data : pd.Series):
    sent = data['sentence']
    sbj = data['subject_word']
    sbj_type = data['subject_type']
    obj = data['object_word']
    obj_type = data['object_type']
		# Subject와 Object에 붙는 문장부호는 다르다!
    sent = sent.replace(sbj, f' @ * {sbj_type} * ' + sbj + ' @ ')
    sent = sent.replace(obj, f' # * {obj_type} * ' + obj + ' # ')
    return sent
# '〈Something〉는 # * PER * 조지 해리슨 #이 쓰고 @ * ORG * 비틀즈 @가 1969년 앨범 《Abbey Road》에 담은 노래다.'