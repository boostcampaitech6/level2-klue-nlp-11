from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from load_data import get_small_data, get_masked_sentences
import copy
import pandas as pd
from py_hanspell.hanspell import spell_checker      ##

model_dir = '/data/ephemeral/dataset/mlm_aug/mlm_aug_model'
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
model = AutoModelForMaskedLM.from_pretrained(model_dir)
dataset_dir = '/data/ephemeral/dataset/mlm_aug/data_en_ja.csv'
new_dataset_dir = '/data/ephemeral/dataset/mlm_aug/mlm_aug_dataset/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
k = 2


def decoding(tokenizer, inputs, original_inputs, predict_mask_topk: dict, k: int):
    
    new_input_ids = copy.deepcopy(inputs.input_ids)[0]
    original_inputs_ids = original_inputs.input_ids[0]
    for mask_idx, predicted_ids in predict_mask_topk.items():
        try:
            original_id = original_inputs_ids[mask_idx]
        except IndexError:
            original_id = [torch.tensor(4)]
        for i in range(k):
            new_id = predicted_ids[i]
            if original_id != new_id:
                new_input_ids[mask_idx] = predicted_ids[i]
                break
    decoded_sentence = tokenizer.decode(new_input_ids, skip_special_tokens=True)
    
    return decoded_sentence

def make_clean_sent(sentence: str):
    sent = sentence.replace('&',' ')
    spelled_sent = spell_checker.check(sent)
    checked_sent = spelled_sent.checked

    return checked_sent

def make_new_row(row, new_sentence: str, sub: str, obj: str, last_df_id: int):
    row2 = row.copy()
    start_sub, start_obj = new_sentence.find(sub), new_sentence.find(obj)
    end_sub, end_obj = start_sub+len(sub)-1, start_obj+len(obj)-1
    row2.sentence, row2.id = new_sentence, last_df_id + 1
    row2.subject_start, row2.subject_end, row2.object_start, row2.object_end = start_sub, end_sub, start_obj, end_obj
    
    return row2

def inference(model, tokenizer, dataset_dir, device):
  
    model.to(device)
    model.eval()
    small_data = get_small_data(dataset_dir)
    new_data = pd.DataFrame()
    last_df_id = small_data.iloc[-1].id
    label_count = {label: len(small_data[small_data.label==label]) for label in small_data.label.unique()}
  
    for id in small_data.id:
        row = small_data[small_data['id']==id].copy()
        original_sentence, sub, obj, label = row.sentence.values[0], row.subject_word.values[0], row.object_word.values[0], row.label.values[0]
        original_inputs = tokenizer(original_sentence, return_tensors='pt')
        original_inputs = original_inputs.to(device)
        masked_sents = get_masked_sentences(tokenizer, original_sentence, sub, obj)
        new_sentences = set()

        for masked_sent in masked_sents:
            inputs = tokenizer(masked_sent, return_tensors='pt')
            inputs = inputs.to(device)
            
        # 모든 마스크 토큰의 위치 찾기
            mask_token_indices = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

            # 모델로부터 마스크 토큰의 예측값 얻기
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            # 각 마스크 토큰에 대한 가장 가능성 높은 토큰 예측
            predict_mask_topk = {}
            for mask_token_index in mask_token_indices:
                predicted_token_ids = predictions[0, mask_token_index].topk(k).indices
                predict_mask_topk[mask_token_index] = predicted_token_ids
         
            decoded_sentence = decoding(tokenizer, inputs, original_inputs, predict_mask_topk, k)
            new_sentences.add(decoded_sentence)

        for new_sent in new_sentences:
            new_sent = make_clean_sent(new_sent)
            if new_sent.find(sub) < 0 or new_sent.find(obj) < 0:
                continue
            new_row = make_new_row(row, new_sent, sub, obj, last_df_id)
            new_data = pd.concat([new_data, pd.DataFrame(new_row)])
            last_df_id += 1
            label_count[label] += 1

    print(f'새로운 데이터 수: {len(new_data)}')
    new_data.drop_duplicates(subset=['sentence', 'label','subject_word','object_word'], inplace=True)
    print(f'중복제거 후 : {len(new_data)}')
    new_data.to_csv(new_dataset_dir+'mlm_aug_data2.csv', encoding='utf-8')
    

def main():
    inference(model, tokenizer, dataset_dir, device)     

if __name__ == '__main__':
    main()