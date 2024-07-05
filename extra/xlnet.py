

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from transformers import MBartForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from src.mbart.configuration_mbart import MBartConfig
from src.mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
from src.mbart.tokenization_mbart import MBartTokenizer

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AlbertTokenizer, AutoTokenizer

from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetLMHeadModel, XLNetConfig, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm



PATH = "datasets/wiki/"

datatypes = os.listdir(PATH)
datatypes.remove(".DS_Store")
datatypes.remove("icl")

def get_contents(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            sentences.append(sentence)
    return sentences

def get_duals(datatypes,lang):
    for i in datatypes:
        x = os.listdir(os.path.join(PATH,i))
        x.remove("domain.txt")
        x.remove("train.eng_Latn")
        if lang in x[0]:
            res_lang = get_contents(os.path.join(PATH,i,x[0]))
            res_en = get_contents(os.path.join(PATH,i,"train.eng_Latn"))
            return {lang:res_lang,"en":res_en}
        
        
LANG1 = "hi"
LANG2 = "tel"

data_lang1 = get_duals(datatypes=datatypes,lang=LANG1)
data_lang2 = get_duals(datatypes=datatypes,lang=LANG2)

data_lang1 = pd.DataFrame(data_lang1)
data_lang2 = pd.DataFrame(data_lang2)

combined = pd.concat([data_lang1[['en', LANG1]], data_lang2[['en', LANG2]]], ignore_index=True)
combined["translation"] = combined[LANG1].fillna('') + combined[LANG2].fillna('')
combined = combined.drop([LANG1,LANG2],axis=1)

multilingual_data = combined

print(combined)


# Annotate the DataFrame
data_lang1['x'] = data_lang1['en'] + ' </s> <2hi>'
data_lang1['y'] = '<2hi> ' + data_lang1['hi'] + ' </s>'


    


tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)


df = data_lang1[:20]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)


import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from src.xlnet.modeling_xlnet import XLNetLMHeadModel
from src.xlnet.configuration_xlnet import XLNetConfig
from tqdm import tqdm
import sacrebleu



# Define the dataset class
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
        
        source_ids = source['input_ids']
        source_mask = source['attention_mask']
        target_ids = target['input_ids']
        target_mask = target['attention_mask']
        
        return {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask
        }

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

# Print tokenizer details
print("Tokenizer length:", len(tokenizer))
print("Tokenizer vocab size:", tokenizer.vocab_size)

# Define the datasets
train_dataset = TranslationDataset(train_df, tokenizer)
val_dataset = TranslationDataset(val_df, tokenizer)
test_dataset = TranslationDataset(test_df, tokenizer)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define the MBart model configuration and initialization
enc_dec_config = MBartConfig(vocab_size=64014, bos_token_id=64000, activation_dropout=0.1, attention_dropout=0.1,
                             encoder_layers=2, decoder_layers=3, pad_token_id=0, eos_token_id=64001)
mbart_model = MBartForConditionalGeneration(config=enc_dec_config)

# Define the XLNet model configuration and initialization
dec_only_config = XLNetConfig(vocab_size=64014, bos_token_id=64000, n_layer=6, pad_token_id=0, eos_token_id=64001)
xlnet_model = XLNetLMHeadModel(config=dec_only_config)

# Define optimizer and scheduler for XLNet
optimizer = AdamW(xlnet_model.parameters(), lr=1e-4)
total_steps = len(train_loader) * 3  # Assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define optimizer and scheduler for MBart
mbart_optimizer = AdamW(mbart_model.parameters(), lr=1e-4)
mbart_total_steps = len(train_loader) * 3  # Assuming 3 epochs
mbart_scheduler = get_linear_schedule_with_warmup(mbart_optimizer, num_warmup_steps=0, num_training_steps=mbart_total_steps)

# Training loop for MBart
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mbart_model.to(device)
xlnet_model.to(device)

def train_epoch(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        optimizer.zero_grad()
        
        input_ids = batch['source_ids'].to(device)
        attention_mask = batch['source_mask'].to(device)
        labels = batch['target_ids'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(data_loader)
    return avg_train_loss

def compute_bleu_score(predictions, references):
    return sacrebleu.corpus_bleu(predictions, [references]).score

def compute_chrf_score(predictions, references):
    score = sacrebleu.corpus_chrf(predictions, [references])
    return score.score

def compute_ter_score(predictions, references):
    ter = sacrebleu.corpus_ter(predictions, [references])
    return ter.score

def eval_epoch(model, data_loader, tokenizer, device, epoch):
    model.eval()
    val_loss = 0
    references = []
    hypotheses = []
    
    comet_scores = []
    chrf_scores = []
    bleu_scores = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Validation Epoch {epoch}"):
            input_ids = batch['source_ids'].to(device)
            attention_mask = batch['source_mask'].to(device)
            labels = batch['target_ids'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            # Generate predictions
            preds = torch.argmax(outputs.logits, dim=-1)
            
            predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            print(predictions,references)
            
            # Compute metrics
            bleu_score = compute_bleu_score(predictions, references)
            comet_score = compute_ter_score(predictions, references)
            chrf_score = compute_chrf_score(predictions, references)
            
            comet_scores.append(comet_score)
            bleu_scores.append(bleu_score)
            chrf_scores.append(chrf_score)
    
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    avg_comet_score = sum(comet_scores) / len(comet_scores)
    avg_chrf_score = sum(chrf_scores) / len(chrf_scores)
            
    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss, avg_bleu_score, avg_comet_score, avg_chrf_score

num_epochs = 3

# Training loop for XLNet
print("Training XLNet model...")
for epoch in range(num_epochs):
    train_loss = train_epoch(xlnet_model, train_loader, optimizer, scheduler, device, epoch)
    val_loss, bleu, comet, chrf = eval_epoch(xlnet_model, val_loader, tokenizer, device, epoch)
    print(f"Epoch {epoch + 1}, XLNet Training loss: {train_loss}, Validation loss: {val_loss}, BLEU: {bleu}, COMET: {comet}, CHRF: {chrf}")

# Training loop for MBart
print("Training MBart model...")
for epoch in range(num_epochs):
    train_loss = train_epoch(mbart_model, train_loader, mbart_optimizer, mbart_scheduler, device, epoch)
    val_loss, bleu, comet, chrf = eval_epoch(mbart_model, val_loader, tokenizer, device, epoch)
    print(f"Epoch {epoch + 1}, MBart Training loss: {train_loss}, Validation loss: {val_loss}, BLEU: {bleu}, COMET: {comet}, CHRF: {chrf}")

# Test evaluation for MBart
test_loss, bleu, comet, chrf = eval_epoch(mbart_model, test_loader, tokenizer, device, "Test")
print(f"Test loss: {test_loss}, BLEU: {bleu}, COMET: {comet}, CHRF: {chrf}")

