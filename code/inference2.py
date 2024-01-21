from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data2 import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
import random
import yaml

from model import *
from util.util import *

def ensemble_inference(model_paths, tokenized_sent, device, model_type, do_sequentialdoublebert=0):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)

    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config2.yaml') as f:
        CFG = yaml.safe_load(f)
    
    # Initialize variables to store predictions and probabilities
    final_output_pred = []
    final_output_prob = []
    
    Tokenizer_NAME = CFG['MODEL_NAME']
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])


    for model_path in model_paths:
        
        model_config = AutoConfig.from_pretrained(model_path)
        test_dataset_dir = CFG['TEST_PATH']

        Tokenizer_NAME = CFG['MODEL_NAME']
        tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
        tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])

        model = SpecialPunctBERT(Tokenizer_NAME, model_config, tokenizer)
        state_dict = torch.load(f'{model_path}/pytorch_model.bin')
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        output_pred = []
        output_prob = []
        
        for data in tqdm.tqdm(dataloader):
            with torch.no_grad():
                if do_sequentialdoublebert:
                    outputs = model(
                        input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device),
                        token_type_ids=data['token_type_ids'].to(device),
                        subject_type=data['subject_type'],
                        object_type=data['object_type'],
                        subject_words=data['subject_words'],
                        object_words=data['object_words'])
                elif model_type == 'entity_special' or model_type == "cls_entity_special" or model_type == "sangmin_entity_special" or model_type == "ko_entity_special":
                    outputs = model(
                        input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device),
                        token_type_ids=data['token_type_ids'].to(device),
                        subject_type=data['subject_type'],
                        object_type=data['object_type'])
                else:
                    outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device),)
                

            if do_sequentialdoublebert:
                logits = outputs
            elif model_type == 'base':
                logits = outputs[0]
            else:
                logits = outputs['logits']
        
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.append(result)
            output_prob.append(prob)
        
        final_output_pred.append(np.concatenate(output_pred))
        final_output_prob.append(np.concatenate(output_prob, axis=0))
    
    # Convert list of arrays to a single array for probabilities
    final_output_prob = np.array(final_output_prob)
    
    # Soft-voting: Average the probabilities across all models for each instance
    avg_probs = np.mean(final_output_prob, axis=0)
    
    # Determine the final predicted label for each instance based on the highest average probability
    final_predictions = np.argmax(avg_probs, axis=-1)
    
    return final_predictions.tolist(), avg_probs.tolist()


def inference(model, tokenized_sent, device, model_type, do_sequentialdoublebert=0):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            if do_sequentialdoublebert:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device),
                    subject_type=data['subject_type'],
                    object_type=data['object_type'],
                    subject_words=data['subject_words'],
                    object_words=data['object_words'])
            elif model_type == 'entity_special' or model_type == "cls_entity_special" or model_type == "sangmin_entity_special" or model_type == "ko_entity_special":
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device),
                    subject_type=data['subject_type'],
                    object_type=data['object_type'])
            else:
                outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device))
                
        if do_sequentialdoublebert:
            logits = outputs
        elif model_type == 'base':
            logits = outputs[0]
        else:
            logits = outputs['logits']
        
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def load_test_dataset(dataset_dir, tokenizer, model_type, discrip, do_sequentialdoublebert=0):
    if do_sequentialdoublebert:
        test_dataset = load_data(dataset_dir, model_type, discrip, do_sequentialdoublebert)
        test_label = list(map(int,test_dataset['label'].values))
        tokenized_test, entity_type, entity_words = sequentialdoublebert_tokenized_dataset(test_dataset, tokenizer, CFG['MODEL_TYPE'])

        return test_dataset['id'], tokenized_test, test_label, entity_type, entity_words
    else:
        if model_type == 'base':
            test_dataset = load_data(dataset_dir, model_type, discrip)
            test_label = list(map(int,test_dataset['label'].values))
            # tokenizing dataset
            tokenized_test = tokenized_dataset(test_dataset, tokenizer)
            return test_dataset['id'], tokenized_test, test_label
        
        elif model_type == 'entity_special':
            test_dataset = load_data(dataset_dir, model_type, discrip)
            test_label = list(map(int,test_dataset['label'].values))
            # tokenizing dataset
            tokenized_test, entity_type = special_tokenized_dataset(test_dataset, tokenizer)
            return test_dataset['id'], tokenized_test, test_label, entity_type
        
        elif model_type == 'entity_punct' or model_type == 'ko_entity_punct':
            test_dataset = load_data(dataset_dir, model_type, discrip)
            test_label = list(map(int,test_dataset['label'].values))

            # max_len = 256

            # tokenizing dataset
            tokenized_test = punct_tokenized_dataset(test_dataset, tokenizer)

            # test_ent_pos_emb = get_entity_position_embedding(tokenizer, tokenized_test['input_ids'])

            # tokenized_test['entity_ids'] = making_entity_pos_emb(test_ent_pos_emb, max_len)


            return test_dataset['id'], tokenized_test, test_label

        elif model_type == "cls_entity_special" or model_type == "ko_entity_special":
            test_dataset = load_data(dataset_dir, model_type, discrip)
            test_label = list(map(int, test_dataset['label'].values))
            # tokenizing dataset
            tokenized_test, entity_type = special_tokenized_dataset(test_dataset, tokenizer)
            return test_dataset['id'], tokenized_test, test_label, entity_type


def main(cnt=None):
    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config2.yaml') as f:
        CFG = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model_paths = ['./best_model_fold0', './best_model_fold1', './best_model_fold2', './best_model_fold3', './best_model_fold4']
    # # final_predictions, final_probs = ensemble_inference(model_paths, tokenized_sent, device, model_type, do_sequentialdoublebert=0)

    
    Tokenizer_NAME = CFG['MODEL_NAME']
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])
    
    # MODEL_NAME = CFG['MODEL_SAVE_DIR']
    # model_config = AutoConfig.from_pretrained(MODEL_NAME)
 

    # model = SpecialPunctBERT(Tokenizer_NAME, model_config, tokenizer)
    # state_dict = torch.load(f'{MODEL_NAME}/pytorch_model.bin')
    # model.load_state_dict(state_dict)
    
    test_dataset_dir = CFG['TEST_PATH']
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], CFG['DISCRIP'])
    Re_test_dataset = REDataset(test_dataset ,test_label)
    model_paths = ['./best_model_tpe']

    # model.to(device)
    

    pred_answer, output_prob = ensemble_inference(model_paths, Re_test_dataset, device, CFG['MODEL_TYPE'], do_sequentialdoublebert=CFG['DO_SEQUENTIALBERTMODEL'])
    pred_answer = num_to_label(pred_answer)
  
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv('./prediction/roberta-large_tpe.csv', index=False)
  
    print('---- Finish! ----')

if __name__ == '__main__':
    seed_everything()
    main()