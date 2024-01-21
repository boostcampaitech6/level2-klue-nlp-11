from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer , EarlyStoppingCallback, AutoModelForSequenceClassification, AutoConfig, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader, RandomSampler

import pytorch_lightning as pl
import torch
import pandas as pd

import wandb
from util.util import *


train_data = pd.read_csv("../dataset/train/train.csv")

train, val = train_dev_split(train_data, 0.2, 42)

train_data = train['sentence'].tolist()
val_data = val['sentence'].tolist()


class LineByLineTextDataset(Dataset):
    def __init__(self,tokenizer,data,block_size):
        encoded_data = tokenizer(data,
                  truncation=True,
                  max_length=block_size)
        self.examples = encoded_data['input_ids']
        self.examples = [{"input_ids": torch.tensor(ex,dtype=torch.long)} for ex in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,i):
        return self.examples[i]
    

def prepare_dataset_for_pretraining(tokenizer,train_input,val_input):
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        data=train_input,
        block_size=512,
    )
    # set mlm task
    # DataCollatorForSOP로 변경시 SOP 사용 가능 (DataCollatorForLanguageModeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15 # 0.3
    )
    eval_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        data=val_input,
        block_size=512,
    )

    return train_dataset, data_collator, eval_dataset

def set_trainer_for_pretraining(
        model,
        data_collator,
        dataset,
        eval_dataset,
        epoch = 10,
        batch_size = 16,
        accumalation_step = 1,):

     # set training args
    training_args = TrainingArguments(
        report_to = 'tensorboard',
        output_dir='./pretraining_robertalarge_output',
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumalation_step,
        evaluation_strategy = 'steps',
        eval_steps=500,
        save_steps=500,
        save_total_limit=1,
        fp16=True,
        load_best_model_at_end=True,
        seed=42,
        save_strategy='steps'
    )


    # set Trainer class for pre-training
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=0.001)]

    )

    return trainer


def pretrain():
    """MLM task 기반 사전학습 진행"""
    # fix a seed
    pl.seed_everything(seed=42)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # set model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large")
    model.to(device)

    # set data
    train_dataset, data_collator, eval_dataset = prepare_dataset_for_pretraining(tokenizer, train_data, val_data)

    # set trainer
    trainer = set_trainer_for_pretraining(model,data_collator,train_dataset,eval_dataset)

    # train model
    print("--- Start train ---")
    trainer.train()
    print("--- Finish train ---")
    model.save_pretrained("./pretrained_roberta_large")


pretrain()