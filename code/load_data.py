import pickle as pickle
import os
import pandas as pd
import torch
import ast
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from transformers import DataCollatorWithPadding
import copy

sub_type_dict = {'PER': '인물', 'ORG': '단체', 'LOC': '장소'}
obj_type_dict = {'PER': '인물', 'ORG': '단체', 'POH': '명칭', 'DAT': '날짜', 'LOC': '장소', 'NOH': '수'}


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)
  

def preprocessing_dataset(dataset):
    sentences = []
    ids = dataset['id']
    labels = dataset['label']

    for sub_entity, obj_entity, sentence, id_, label in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence'], ids, labels):
        sub_entity = eval(sub_entity)
        obj_entity = eval(obj_entity)

        sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
        sub_type, obj_type = sub_entity['type'], obj_entity['type']
        sub_word, obj_word = f" '{sub_entity['word']}' ", f" '{obj_entity['word']}' "

        sub_marker = f"@ § {sub_type_dict[sub_type]} §{sub_word}@"
        obj_marker = f"# ^ {obj_type_dict[obj_type]} ^{obj_word}#"

        if sub_idx[0] < obj_idx[0]:
            parts = [sentence[:sub_idx[0]], sub_marker, sentence[sub_idx[1] + 1:obj_idx[0]],
                     obj_marker, sentence[obj_idx[1] + 1:]]
        else:
            parts = [sentence[:obj_idx[0]], obj_marker, sentence[obj_idx[1] + 1:sub_idx[0]],
                     sub_marker, sentence[sub_idx[1] + 1:]]

        sentence = ''.join(parts)
        sentence = add_query(sentence, sub_word, obj_word, f" '{obj_type}'", f" '{obj_type}'")

        sentences.append(sentence)

    out_dataset = pd.DataFrame({'id': ids, 'sentence': sentences, 'label': labels})

    return out_dataset


def load_data(dataset_dir):
  
  dataset_pd = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(dataset_pd)

  return dataset

def add_query(sentence, sub_word, obj_word, sub_type, obj_type):
    query = f"{obj_word}는{sub_word}의{obj_type}이다."

    return f"{sentence}:{query}"

def tokenized_dataset(dataset, tokenizer):
  tokenized_sentences = tokenizer(
    list(dataset['sentence']),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    add_special_tokens=True)
  
  return tokenized_sentences
