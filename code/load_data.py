import pickle as pickle
import os
import pandas as pd
import torch
import ast
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

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

MARKERS_TO_KOR = {
  "PER" : "사람", "ORG" : "단체", "LOC" : "장소", "DAT" : "날짜", "POH" : "수량", "NOH" : "기타"
}


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
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  subject_type = []
  object_type = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    # a = i[1:-1].split(',')[0].split(':')[1].strip().strip("'")
    # b = j[1:-1].split(',')[0].split(':')[1].strip().strip("'")
    # c = i[1:-1].split(',')[3].split(':')[1].strip().strip("'")
    # d = j[1:-1].split(',')[3].split(':')[1].strip().strip("'")
    subject_entity_dict = ast.literal_eval(i)
    object_entity_dict = ast.literal_eval(j)
    a = subject_entity_dict['word']
    b = object_entity_dict['word']
    c = subject_entity_dict['type']
    d = object_entity_dict['type']

    
    subject_entity.append(a)
    object_entity.append(b)
    subject_type.append(c)
    object_type.append(d)
  out_dataset = pd.DataFrame(
    {
    'id':dataset['id'], 
    'sentence':dataset['sentence'],
    'subject_entity':subject_entity, 
    'subject_type': subject_type, 
    'object_entity':object_entity,
    'object_type': object_type, 
    'label':dataset['label'],
    }
    )
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, e03, e04 in zip(dataset['subject_entity'], dataset['object_entity'], dataset['subject_type'], dataset['object_type']):
    temp = ''
    # temp = e01 +' 와 '+ e02 +' 의 관계는 '+ MARKERS_TO_KOR[e03] +' 와 '+ MARKERS_TO_KOR[e04] +'의 관계이다.'
    temp = f"{e01}와 {e02} 의 관계는 {MARKERS_TO_KOR[e03]}와 {MARKERS_TO_KOR[e04]}의 관계이다."

    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences



def train_dev_split(dataset, ratio: float, random_state: int = 42):
    if ratio == 0.0:
        ratio = 0.1

    label_dict = defaultdict(int)

    for label in dataset['label']:
        label_dict[label] += 1

    train_dataset = []
    dev_dataset = []

    for item in tqdm(label_dict.items(), desc='train_dev_split', total=len(label_dict)):
        sub_dataset = dataset[dataset['label'] == item[0]]

        train_data, dev_data = train_test_split(sub_dataset, test_size=ratio, random_state=random_state)

        train_dataset.append(train_data)
        dev_dataset.append(vdev_data)

    train_dataset = pd.concat(train_dataset, ignore_index=True)
    dev_dataset = pd.concat(dev_dataset, ignore_index=True)

    train_dataset = train_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    dev_dataset = dev_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_dataset, dev_dataset