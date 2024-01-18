import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm
from utils.utils import entity_marker, typed_entity_marker, typed_entity_marker_punc, TYPE_MARKERS_PUNC

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
    typed_sentence = []
    for i, data in dataset.iterrows():
        typed_sentence.append(typed_entity_marker_punc(data))
        # print(data['sentence'])
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':typed_sentence, 'subject_entity':dataset['subject_word'], 'object_entity':dataset['object_word'], 'label':dataset['label'],})
    return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        # Prompting Sentence 
        temp = ''
        temp = e01 + '와(과)' + e02 + '은(는)?'# +'[SEP]'
        concat_entity.append(temp)
        
    # num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens" : list(TYPE_MARKERS_PUNC.values())})
    # print("We have added", num_added_toks, "tokens")

    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
    return tokenized_sentences

def get_entity_position_embedding(tokenizer, input_ids):
    special_token2id = {k:v for k,v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}

    sub_token_id = special_token2id['@']
    obj_token_id = special_token2id['#']
    
    pos_embeddings = []

    for y in tqdm(input_ids):
        pos = []
        for j in range(0, len(y)):
            if len(pos) == 4:
                break
            if y[j] == sub_token_id:
                pos.append(j)

            if y[j] == obj_token_id:
                pos.append(j)
        pos_embeddings.append(pos)
    return pos_embeddings

def making_entity_pos_emb(pos_emb, max_length=256):
    ddd = []
    for idx, ids in tqdm(enumerate(pos_emb)):
        ent_emb = []
        ent_emb += [0] * ids[0] +\
                   [1] * (ids[1] - ids[0] + 1) + \
                   [0] * (ids[2] - ids[1] - 1) + \
                   [1] * (ids[3] - ids[2] + 1) + \
                   [0] * (max_length - ids[3]-1)
        ddd.append(ent_emb)
    return torch.Tensor(ddd)


# def entity_ids_maker(data, start_id, end_id): # data에는 tokenizer를 거쳐 나온 input_ids가 들어온다
#     def update_ranges_to_1(start_tokens, end_tokens, maxlen=251):
#         # print(start_tokens)
#         # print(end_tokens)
#         res = []
#         res += [0] * (start_tokens[0]+1) + [1] * (end_tokens[0]-start_tokens[0]-1)\
#         + [0] * (start_tokens[1]-end_tokens[0]+1)\
#         + [1] * (end_tokens[1]-start_tokens[1]-1)\
#         + [0] * (maxlen-end_tokens[1])
#         return res
             
#     entity_ids = []
#     for ids in tqdm(data):
#         startidx = []
#         endidx = []
        
#         for i in range(len(ids)):
#             if ids[i] in start_id:
#                 # print(tokens[i])
#                 startidx.append(i)
                
#             elif ids[i] in end_id:
#                 # print(tokens[i])
#                 endidx.append(i)
                
#         tmp = update_ranges_to_1(startidx, endidx, maxlen=len(ids))
        
#         entity_ids.append(tmp)
#     entity_ids = torch.Tensor(entity_ids)
#     return entity_ids # Tensor로 변환하기