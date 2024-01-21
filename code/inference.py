from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
from tqdm import tqdm
import yaml
from pytorch_lightning import seed_everything

from load_data import *
from model import *
from util.util import *

def ensemble_inference(model_paths, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)

    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Initialize variables to store predictions and probabilities
    final_output_pred = []
    final_output_prob = []
    
    Tokenizer_NAME = config['MODEL_NAME']
    # tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    # tokenizer = add_token(tokenizer)


    for model_path in model_paths:
        
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.num_labels = 30

        Tokenizer_NAME = config['MODEL_NAME']
        tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
        tokenizer = add_token(tokenizer)

        model = CustomModel(Tokenizer_NAME, model_config, tokenizer)
        state_dict = torch.load(f'{model_path}/pytorch_model.bin')
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        output_pred = []
        output_prob = []
        
        for data in tqdm(dataloader):
            with torch.no_grad():
                outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device),)
                

            logits = outputs['logits']
        
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.append(result)
            output_prob.append(prob)
        
        final_output_pred.append(np.concatenate(output_pred))
        final_output_prob.append(np.concatenate(output_prob, axis=0))
    
    final_output_prob = np.array(final_output_prob)
    
    avg_probs = np.mean(final_output_prob, axis=0)
    
    final_predictions = np.argmax(avg_probs, axis=-1)
    
    return final_predictions.tolist(), avg_probs.tolist()


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device))
                
        logits = outputs['logits']
        
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def load_test_dataset(dataset_dir, tokenizer):

    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))

    tokenized_test = tokenized_dataset(test_dataset, tokenizer)

    return test_dataset['id'], tokenized_test, test_label


def main(cnt=None):
    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config.yaml') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    Tokenizer_NAME = config['MODEL_NAME']
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    tokenizer = add_token(tokenizer)
    
    test_dataset_dir = config['TEST_PATH']
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label)
    model_paths = config["MODEL_PATHS"]

    pred_answer, output_prob = ensemble_inference(model_paths, Re_test_dataset, device)
    pred_answer = num_to_label(pred_answer)
  
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(config['INFERENCE_OUTPUT_PATH'], index=False)
  
    print('---- Finish! ----')

if __name__ == '__main__':
    with open('/data/ephemeral/lost+found/level2-klue-nlp-11/code/config.yaml') as f:
        config = yaml.safe_load(f)

    seed_everything(config['SEED'])
    main()