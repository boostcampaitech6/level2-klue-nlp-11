import pandas as pd
import torch
import pytorch_lightning as pl
import transformers
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import pickle
import numpy as np
import random

import yaml
config = yaml.load(open('./config.yaml', 'r'), Loader = yaml.Loader)
sweep = yaml.load(open('./sweep.yaml', 'r'), Loader = yaml.Loader)

from traindevsplit import *
# from preprocessing import *

MARKERS = dict(
    subject_start_marker="<SUB>",
    subject_end_marker  ="</SUB>",
    object_start_marker ="<OBJ>",
    object_end_marker   ="</OBJ>",
)
TYPE_MARKERS = dict(
    subject_start_per_marker="<S:PER>",
    subject_start_org_marker="<S:ORG>",
    subject_end_per_marker ="</S:PER>",
    subject_end_org_marker ="</S:ORG>",
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

def set_seed(seed:int = config['seed']):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class REDataset(torch.utils.data.Dataset):
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

class Dataloader(pl.LightningDataModule):

    def __init__(
            self,
            model_name,
            batch_size,
            num_workers,
            shuffle,
            train_path,
            dev_path,
            test_path,
            predict_path,
            sep_token = "None",
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path


        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.sep_token = sep_token

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        if sep_token != "None":
            if sep_token == "EntityMarker":
                self.tokenizer.add_tokens(list(MARKERS.values()))
            elif sep_token == "TypedEntityMarker":
                self.tokenizer.add_tokens(list(TYPE_MARKERS.values()))
            elif sep_token == "TypedEntityMarkerPunt":
                self.tokenizer.add_tokens(["*","@","#"])

        def prepare_data(self):
            self.train_dataframe = pd.read_csv(self.train_path)
            self.dev_dataframe = pd.read_csv(self.dev_path)
            self.test_dataframe = pd.read_csv(self.test_path)
            self.predict_dataframe = pd.read_csv(self.predict_path)

            self.train_sents, self.train_labels, self.train_sub, self.train_obj = self.preprocessing(self.train_dataframe)
            self.dev_sents, self.dev_labels, self.dev_sub, self.dev_obj = self.preprocessing(self.dev_dataframe)
            self.test_sents, self.test_labels, self.test_sub, self.test_obj = self.preprocessing(self.test_dataframe)
            self.train_sents, self.train_labels, self.train_sub, self.train_obj = self.preprocessing(self.train_dataframe)
        
        def preprocessing(self, dataset):

            dataset = self.tokenize(dataset)

            sents = dataset['tokenized_sent']
            labels = dataset['label']
            sub = dataset['subject_word']
            obj = dataset['object_word']

            return sents, labels, sub, obj



        def tokenize(self, dataset):

            for idx, sents in tqdm(dataset.iterrows(), desc='tokenizeing', total = len(dataset)):

                sent = sents['sentence']
                sbj = sents['subject_word']
                sbj_start_type_mark = TYPE_MARKERS[f"subject_start_{sents['subject_type'].lower()}_marker"]
                sbj_end_type_mark = TYPE_MARKERS[f"subject_end_{sents['subject_type'].lower()}_marker"]
                obj = sents['object_word']
                obj_start_type_mark = TYPE_MARKERS[f"object_start_{sents['object_type'].lower()}_marker"]
                obj_end_type_mark = TYPE_MARKERS[f"object_end_{sents['object_type'].lower()}_marker"]
                sent = sent.replace(sbj, sbj_start_type_mark+' '+sbj+' '+sbj_end_type_mark)
                sent = sent.replace(obj, obj_start_type_mark+' '+obj+' '+obj_end_type_mark)

                tokenized_sent = self.tokenizer(
                    sent,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                    add_special_tokens=True,
                )

                dataset.loc[idx, 'tokenized_sent'] = tokenized_sent
                dataset.loc[idx, 'tokenized_sent_len'] = len(tokenized_sent['input_ids'][0])

            return dataset

        def setup(self, stage='fit'):
            if stage == 'fit':
                self.train_dataset = REDataset(self.train_sents, self.train_labels)
                self.dev_dataset = REDataset(self.dev_sents, self.dev_labels)
            elif stage == 'test':
                self.test_dataset = REDataset(self.test_sents, self.test_labels)
            elif stage == 'predict':
                self.predict_dataset = REDataset(self.predict_sents, self.predict_labels)
            
            
        
        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
        
        def dev_dataloader(self):
            return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

        def test_dataloader(self):
            return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
        
        def predict_dataloader(self):
            return torch.utils.data.DataLoader(self.prediction_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)




        





##아래는일단무시

# def preprocessing_dataset(dataset):
#     subject_entity = []
#     object_entity = []
#     for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
#         i = i[1:-1].split(',')[0].split(':')[1]
#         j = j[1:-1].split(',')[0].split(':')[1]

#         subject_entity.append(i)
#         object_entity.append(j)
#     out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
#     return out_dataset

# def load_data(dataset_dir):
#     pd_dataset = pd.read_csv(dataset_dir)
#     dataset = preprocessing_dataset(pd_dataset)
    
#     return dataset

# def tokenized_dataset(dataset, tokenizer, tokenizing_option):
#     concat_entity = []

#     if tokenizing_option == 'base':
#         concat_entity = []
#         for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
#             temp = ''
#             temp = e01 + '[SEP]' + e02
#             concat_entity.append(temp)

#         tokenized_sentences = tokenizer(
#         concat_entity,
#         list(dataset['sentence']),
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=256,
#         add_special_tokens=True,
#         )

#     elif tokenizing_option == 'typed_entity_marker':

#             sent = dataset['sentence']
#             sbj = dataset['subject_word']
#             sbj_start_type_mark = TYPE_MARKERS[f"subject_start_{dataset['subject_type'].lower()}_marker"]
#             sbj_end_type_mark = TYPE_MARKERS[f"subject_end_{dataset['subject_type'].lower()}_marker"]
#             obj = dataset['object_word']
#             obj_start_type_mark = TYPE_MARKERS[f"object_start_{dataset['object_type'].lower()}_marker"]
#             obj_end_type_mark = TYPE_MARKERS[f"object_end_{dataset['object_type'].lower()}_marker"]
#             sent = sent.replace(sbj, sbj_start_type_mark+' '+sbj+' '+sbj_end_type_mark)
#             sent = sent.replace(obj, obj_start_type_mark+' '+obj+' '+obj_end_type_mark)




#     return tokenized_sentences