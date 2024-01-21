from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import BertModel, BertConfig, BertForSequenceClassification, BertForMaskedLM

from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from train import set_seed

import yaml
config = yaml.load(open('./config.yaml', 'r'), Loader = yaml.Loader)

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  set_seed(config['seed'])
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  # load tokenizer
  Tokenizer_NAME = config['MODEL_NAME']
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  # ## load my model
  # MODEL_NAME = args.model_dir # model dir.
  # model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  # model.parameters
  # model.to(device)
  # Load models
  models = []
  for model_dir in config['model_dirs']:  # Assuming config['model_dirs'] is a list of model directories
      model = AutoModelForSequenceClassification.from_pretrained(model_dir)
      model.to(device)
      model.eval()
      models.append(model)

  ## load test datset
  test_dataset_dir = config['test']['test_dataset_filepath']
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  # pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  # pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  # Predict answer using ensemble
  pred_answer, output_prob = ensemble_soft(models, Re_test_dataset, device)
  pred_answer = num_to_label(pred_answer)
  # For hard voting, the call would be slightly different
  # pred_answer = hard_voting_inference(models, Re_test_dataset, device)


  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv(config['test']['predict_submission_filepath'], index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('saved csv file to {}'.format(config['test']['predict_submission_filepath']))
  print('---- Finish! ----')

def ensemble_soft(models, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    final_output_pred = []
    final_output_prob = []

    for i, data in enumerate(tqdm(dataloader)):
        all_logits = None
        for j, model in enumerate(models):
            with torch.no_grad():
                outputs = model(input_ids=data['input_ids'].to(device),
                                attention_mask=data['attention_mask'].to(device),
                                token_type_ids=data['token_type_ids'].to(device))
                logits = outputs.logits

                if all_logits is None:
                    all_logits = logits
                else:
                    all_logits += logits

        avg_logits = all_logits / len(models)
        avg_prob = F.softmax(avg_logits, dim=-1).detach().cpu().numpy()
        avg_logits = avg_logits.detach().cpu().numpy()
        avg_result = np.argmax(avg_logits, axis=-1)

        final_output_pred.append(avg_result)
        final_output_prob.append(avg_prob)

    return np.concatenate(final_output_pred).tolist(), np.concatenate(final_output_prob, axis=0).tolist()

def hard_voting_inference(models, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    final_output_pred = []

    for i, data in enumerate(tqdm(dataloader)):
      predictions = []

      # Collect predictions from each model
      for j, model in models:
          with torch.no_grad():
              outputs = model(input_ids=data['input_ids'].to(device),
                              attention_mask=data['attention_mask'].to(device),
                              token_type_ids=data['token_type_ids'].to(device))
              logits = outputs.logits
              result = np.argmax(logits.detach().cpu().numpy(), axis=-1)
              predictions.append(result)
              # print("Model {} finished predicting".format(j))

      # Transpose to get predictions per instance across models
      predictions = np.array(predictions).T

      # Hard voting: take the mode (most common prediction) across models
      mode_pred = [np.bincount(pred).argmax() for pred in predictions]
      final_output_pred.extend(mode_pred)

      return final_output_pred


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default=config['best_model_dir'])
  args = parser.parse_args()
  print(args)
  main(args)
  
