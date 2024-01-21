from transformers import AutoModel, BertPreTrainedModel, AutoModelForSequenceClassification, AutoConfig
import torch
from copy import deepcopy
import numpy as np

from load_data import *
from util.util import *

class CustomModel(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)
        self.calculated_token_count = 2
        
        # Initialize the entity embedding layer
        # self.entity_embedding = torch.nn.Embedding(2, config.hidden_size)  # 2 for entity and non-entity

        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))

        self.tokenizer = tokenizer
        self.config = config
        
        ids = tokenizer.convert_tokens_to_ids(['@', '#'])
        self.sub_ids, self.obj_ids = ids[0], ids[1]
        
        self.classifier = torch.nn.Sequential(
            self.model.pooler,
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True))
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(self.calculated_token_count)])
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[1/self.calculated_token_count]] for _ in range(self.calculated_token_count)]))
        
        self.loss_function = LabelSmoothingLoss()  # Initialize loss function here for reuse

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, entity_ids=None, output_attentions=False):
        model_outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = model_outputs.last_hidden_state

        special_indices = self.find_special_token_indices(input_ids)
        pooled_outputs = self.pool_special_outputs(special_outputs, special_indices)
        
        logits = self.compute_logits(pooled_outputs)
        logits = torch.sum(self.weight_parameter * logits, dim=0)
        
        loss = self.compute_loss(logits, labels) if labels is not None else None
        
        return self.format_outputs(loss, logits, model_outputs.attentions if output_attentions else None)

    # def find_special_token_indices(self, input_ids):
    #     special_indices = []
    #     for idx in input_ids:
    #         sub_idx = (idx == self.sub_ids).nonzero(as_tuple=True)[0]
    #         obj_idx = (idx == self.obj_ids).nonzero(as_tuple=True)[0]
    #         special_indices.append([sub_idx, obj_idx])
    #     return torch.stack(special_indices)
    
    # def find_special_token_indices(self, input_ids):
    #     special_indices = []
    #     for idx in input_ids:
    #         sub_idx = (idx == self.sub_ids).nonzero(as_tuple=True)[0]
    #         obj_idx = (idx == self.obj_ids).nonzero(as_tuple=True)[0]
    #         # Ensure that sub_idx and obj_idx are tensors, not lists
    #         special_indices.append(torch.tensor([sub_idx, obj_idx], device=input_ids.device))
    #     # Now special_indices is a list of tensors, so torch.stack should work
    #     return torch.stack(special_indices)
    
    def find_special_token_indices(self, input_ids):
        special_indices = []
        for idx in input_ids:
            sub_idx = (idx == self.sub_ids).nonzero(as_tuple=True)[0]
            obj_idx = (idx == self.obj_ids).nonzero(as_tuple=True)[0]
            
            # Ensure that we are only taking the first occurrence if there are multiple.
            # Use .item() to extract the scalar value for indexing.
            if sub_idx.nelement() != 0 and obj_idx.nelement() != 0:
                sub_idx = sub_idx[0].item()
                obj_idx = obj_idx[0].item()
                special_indices.append([sub_idx, obj_idx])
            else:
                # Handle the case where the special tokens are not found
                # You might want to handle this case separately based on your use-case
                special_indices.append([0, 0])  # Placeholder, you might want to handle this differently

        # Now special_indices is a list of lists, so we convert it to tensor
        return torch.tensor(special_indices, dtype=torch.long, device=input_ids.device)
    
    def pool_special_outputs(self, special_outputs, special_indices):
        return [torch.stack([special_outputs[i, idx,:] for i, idx in enumerate(special_indices[:, j])]) for j in range(self.calculated_token_count)]

    def compute_logits(self, pooled_outputs):
        return torch.stack([self.special_classifier[i](pooled_output.unsqueeze(1)) for i, pooled_output in enumerate(pooled_outputs)], dim=0)

    def compute_loss(self, logits, labels):
        return self.loss_function(logits.view(-1, self.config.num_labels), labels.view(-1))

    def format_outputs(self, loss, logits, attentions):
        outputs = {"loss": loss, "logits": logits}
        if attentions is not None:
            outputs["attentions"] = attentions[0]
        return outputs
        
# class CustomModel1(BertPreTrainedModel):
#     def __init__(self, model_name, config, tokenizer):
#         super().__init__(config)
#         self.calculated_token_count = 2
        
#         # Initialize the entity embedding layer
#         # self.entity_embedding = torch.nn.Embedding(2, config.hidden_size)  # 2 for entity and non-entity

#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.resize_token_embeddings(len(tokenizer))

#         self.tokenizer = tokenizer
#         self.config = config
        
#         ids = tokenizer.convert_tokens_to_ids(['@', '#'])
#         self.sub_ids, self.obj_ids = ids[0], ids[1]
        
#         self.classifier = torch.nn.Sequential(
#             self.model.pooler,
#             torch.nn.Dropout(p=0.1),
#             torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True))
        
#         self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(self.calculated_token_count)])
#         self.weight_parameter = torch.nn.Parameter(torch.tensor([[[1/self.calculated_token_count]] for _ in range(self.calculated_token_count)]))
        
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, entity_ids=None, output_attentions=False):
#         outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
#         special_outputs = outputs.last_hidden_state

#         batch_size = len(input_ids)
#         special_idx = list()
        
#         for i in range(batch_size):
#             sub_start_idx = torch.nonzero(input_ids[i] == self.sub_ids)[0][0]
#             obj_start_idx = torch.nonzero(input_ids[i] == self.obj_ids)[0][0]
            
#             special_idx.append([sub_start_idx, obj_start_idx])
        
#         pooled_output = [torch.stack([special_outputs[i, special_idx[i][j],:] for i in range(batch_size)]) for j in range(self.calculated_token_count)]

#         logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(self.calculated_token_count)], dim=0)
#         logits = torch.sum(self.weight_parameter * logits, dim=0) 
    
#         loss = None
        
#         loss_fun = LabelSmoothingLoss()
#         if labels is not None:  ##
#             # Ensure labels are not None before computing loss
#             loss_fun = LabelSmoothingLoss()
#             loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))

#         if output_attentions:    
#             outputs = {"loss": loss, "logits": logits, "attentions": outputs.attentions[0]}
#         else:
#             outputs = {"loss": loss, "logits": logits}
        
#         return outputs

# class SpecialPunctBERT(BertPreTrainedModel):
#     def __init__(self, model_name, config, tokenizer):
#         super().__init__(config)

#         self.calculated_token_count = 2

#         # self.entitiy_embedding = torch.nn.Embedding(2, config.hidden_size)#entityembedding
        
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.resize_token_embeddings(len(tokenizer))
        
#         self.tokenizer = tokenizer
#         self.config = config
        
#         ids = tokenizer.convert_tokens_to_ids(['@', '#'])
#         self.sub_ids, self.obj_ids = ids[0], ids[1]
        
#         self.classifier = torch.nn.Sequential(
#             self.model.pooler,
#             torch.nn.Dropout(p=0.1),
#             torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True))
        
#         self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(self.calculated_token_count)])
#         self.weight_parameter = torch.nn.Parameter(torch.tensor([[[1/self.calculated_token_count]] for _ in range(self.calculated_token_count)]))
        
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, output_attentions=False):
#         outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
#         special_outputs = outputs.last_hidden_state

#         batch_size = len(input_ids)
#         special_idx = list()
        
#         for i in range(batch_size):
#             sub_start_idx = torch.nonzero(input_ids[i] == self.sub_ids)[0][0]
#             obj_start_idx = torch.nonzero(input_ids[i] == self.obj_ids)[0][0]
            
#             special_idx.append([sub_start_idx, obj_start_idx])
        
#         pooled_output = [torch.stack([special_outputs[i, special_idx[i][j],:] for i in range(batch_size)]) for j in range(self.calculated_token_count)]

#         logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(self.calculated_token_count)], dim=0)
#         logits = torch.sum(self.weight_parameter * logits, dim=0) 
    
#         loss = None
        
#         if labels is not None:
#             loss_fun = torch.nn.CrossEntropyLoss()
#             loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
         
#         if output_attentions:    
#             outputs = {"loss": loss, "logits": logits, "attentions": outputs.attentions[0]}
#         else:
#             outputs = {"loss": loss, "logits": logits}
        
#         return outputs
 