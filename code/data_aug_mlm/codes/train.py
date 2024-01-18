from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from load_data import load_data, tokenize

model_name = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
dataset_dir = '/data/ephemeral/dataset/mlm_aug/data_en_ja.csv'

def train(model, inputs, labels):
  
  # 데이터셋 및 데이터 로더 생성
  dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, labels)
  dataloader = DataLoader(dataset, batch_size=10)
  
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  # 모델 학습을 위한 설정
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
  model.train()

  # 학습 반복문
  for epoch in range(20):  
      for batch in dataloader:
          optimizer.zero_grad()
          input_ids, attention_mask, labels = [b.to(model.device) for b in batch]
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss
          loss.backward()
          optimizer.step()

      print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")
      
  model.save_pretrained('/data/ephemeral/dataset/mlm_aug/mlm_aug_model')

def main():
  original_sentences, masked_sentences = load_data(tokenizer, dataset_dir)
  inputs, labels = tokenize(tokenizer, original_sentences, masked_sentences)
  train(model, inputs, labels)

if __name__ == '__main__':
  main()
