from transformers import DataCollatorForLanguageModeling
import torch
import pandas as pd

def get_small_data(dataset_dir):
  data = pd.read_csv(dataset_dir)
  small_label_df = pd.DataFrame()
  for label in data.label.unique():
    df = data[data['label'] == label]
    if len(df) < 400:
      small_label_df = pd.concat([small_label_df, df])
  
  return small_label_df

def get_original_sentences(df):
  sentences_list = [sent for sent in df.sentence]

  return sentences_list

def get_sub_obj(df):
  sub_list = [sub for sub in df.subject_word]
  obj_list = [obj for obj in df.object_word]

  return sub_list, obj_list

def tokenize(tokenizer, original_sents: list, masked_sents: list):  # 토큰화 및 텐서 변환
  inputs = tokenizer(masked_sents, padding=True, truncation=True, return_tensors="pt")
  labels = tokenizer(original_sents, padding=True, truncation=True, return_tensors="pt").input_ids
  
  return inputs, labels


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


def load_data(tokenizer, dataset_dir):
    small_data = get_small_data(dataset_dir)
    original_sents = get_original_sentences(small_data)
    sub_words, obj_words = get_sub_obj(small_data)
    original_sentences, masked_sentences = [], []
    cnt = 0

    for sent, sub, obj in zip(original_sents, sub_words, obj_words):
    
        masked_sents = get_masked_sentences(tokenizer, sent, sub, obj)
        cnt += 1
        if len(masked_sents) == 0:
            continue
        original_sentences.append(sent)
        masked_sentences.append(masked_sents[0])
    print(len(original_sentences), len(masked_sentences))

    return original_sentences, masked_sentences


