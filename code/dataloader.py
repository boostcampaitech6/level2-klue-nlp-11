import pandas as pd
import torch
import pytorch_lightning as pl
import transformers
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, subject)

