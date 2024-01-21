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
from transformers import DataCollatorForLanguageModeling

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


def get_masked_sentences(tokenizer, text: str, sub_word: str, obj_word: str):   # Data Collator를 사용하여 문장 랜덤 마스킹

  inputs = tokenizer(text, return_tensors="pt")

  
  do_not_mask_words = [sub_word, obj_word, '[CLS]', '[SEP]']
  protected_tokens = tokenizer(do_not_mask_words, add_special_tokens=False)['input_ids']
  protected_tokens = [item for sublist in protected_tokens for item in sublist]   # + ending_words_id_list # Flatten the list

  masked_sentences = []   
  for j in range(10):   # 한 문장에 대하여 랜덤 마스킹 여러 번 수행 (다양한 버전을 얻기 위함)
    if len(masked_sentences) >= 5:
      break
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)   #, mlm_probability=0.3)   # Data Collator 초기화
    masked_inputs = data_collator([inputs])   # Data Collator를 사용하여 마스킹 적용
    
    # protected_tokens는 마스킹에서 제외
    for i, input_id in enumerate(inputs['input_ids'][0]):
      if any([i >= len(masked_inputs['input_ids'][0][0])-4, input_id.item() in protected_tokens]):
          masked_inputs['input_ids'][0][0][i] = input_id    

    # 디코딩 및 문장 깨끗하게 정리
    output = tokenizer.decode(masked_inputs['input_ids'][0][0])
    output = output.replace('[CLS]','').replace('[SEP]','')
    output = output.strip()
    if output.find(sub_word) < 0 or output.find(obj_word) < 0:
      continue
    masked_sentences.append(output)
    
  return masked_sentences