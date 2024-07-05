
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_max_len=128, target_max_len=128):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        source_text = str(self.dataframe.iloc[index]['x'])
        target_text = str(self.dataframe.iloc[index]['y'])
        
        source = self.tokenizer(source_text, padding='max_length', truncation=True, max_length=self.source_max_len, return_tensors="pt")
        target = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=self.target_max_len, return_tensors="pt")
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        return {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'source_text':source_text,
            'target_text':target_text
        }
        
        