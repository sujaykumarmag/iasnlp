

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sacrebleu

PATH = "datasets/wiki/"

datatypes = os.listdir(PATH)
datatypes.remove(".DS_Store")

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
LANG2 = "mar"

data_lang1 = get_duals(datatypes=datatypes,lang=LANG1)
data_lang2 = get_duals(datatypes=datatypes,lang=LANG2)

data_lang1 = pd.DataFrame(data_lang1)
data_lang2 = pd.DataFrame(data_lang2)

combined = pd.concat([data_lang1[['en', LANG1]], data_lang2[['en', LANG2]]], ignore_index=True)
combined["translation"] = combined[LANG1].fillna('') + combined[LANG2].fillna('')
combined = combined.drop([LANG1,LANG2],axis=1)

multilingual_data = combined

print(combined)




import pandas as pd
from transformers import MBartForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


import torch.nn as nn



from mbart.configuration_mbart import MBartConfig
from mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
from mbart.tokenization_mbart import MBartTokenizer

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AlbertTokenizer, AutoTokenizer

from sacrebleu import corpus_bleu, corpus_ter, corpus_chrf
from rouge_score import rouge_scorer


# Annotate the DataFrame
data_lang1['x'] = data_lang1['en'] + ' </s> <2hi>'
data_lang1['y'] = '<2hi> ' + data_lang1['hi'] + ' </s>'


df = data_lang1[:40]

# Print the DataFrame to check the annotated data
print(df[['x', 'y']])

# Tokenizer and Model Configuration
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
enc_dec_config = MBartConfig(vocab_size=64014, bos_token_id=64000, activation_dropout=0.1, attention_dropout=0.1,
                             encoder_layers=2, decoder_layers=3, pad_token_id=0, eos_token_id=64001)
model = MBartForConditionalGeneration(config=enc_dec_config)

# Dataset Class
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
            'target_mask': target_mask
        }

train_df = df.iloc[30:]  # Example, adjust based on your actual data size
val_df = df.iloc[:10]    # Example, adjust based on your actual data size
print(train_df,val_df)

train_dataset = TranslationDataset(train_df, tokenizer)
val_dataset = TranslationDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluation Metrics Functions
def compute_bleu_score(predictions, references):
    return corpus_bleu(predictions, [references]).score



def compute_chrf_score(predictions, references):
    score = corpus_chrf(predictions, [references])
    return score.score


def compute_ter_score(predictions, references):
    ter = corpus_ter(predictions, [references])
    return ter.score



# Training Loop with Evaluation Metrics
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
epochs = 3


metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'TER', 'chrf','bleu'])


for epoch in range(epochs):
    model.train()
    total_loss = 0.0
  
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['source_ids']
        attention_mask = batch['source_mask']
        labels = batch['target_ids']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
  
    # Compute metrics on validation set after each epoch
    model.eval()
    comet_scores = []
    chrf_scores = []
    bleu_scores = []
  
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['source_ids']
            attention_mask = batch['source_mask']
            labels = batch['target_ids']
            
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)
            
            # Decode generated sequences and references
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
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
  
    # Store metrics in DataFrame
    epoch_metrics = {
        'epoch': epoch + 1,
        'train_loss': total_loss,
        'TER': avg_comet_score,
        'chrf': avg_chrf_score,
        'bleu': avg_bleu_score,
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)
  
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss}, Avg COMET: {avg_comet_score}, Avg CHRF: {avg_chrf_score}')
  
# Save metrics DataFrame to a CSV file
metrics_df.to_csv('training_metrics.csv', index=False)
  
print(metrics_df)


# Plotting the metrics
# Use an available style
  
# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(12, 8))
  
# Define the x-axis (epochs)
epochs = metrics_df['epoch']
  
# Plot training loss
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Loss', color='tab:red')
ax1.plot(epochs, metrics_df['train_loss'], label='Train Loss', color='tab:red', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:red')
  
# Create a second y-axis for the BLEU score, TER score, and CHRF score
ax2 = ax1.twinx()
ax2.set_ylabel('Scores', color='tab:blue')
ax2.plot(epochs, metrics_df['bleu'], label='BLEU Score', color='tab:blue', marker='x')
ax2.plot(epochs, metrics_df['TER'], label='TER Score', color='tab:orange', marker='s')
ax2.plot(epochs, metrics_df['chrf'], label='CHRF Score', color='tab:green', marker='^')
  
fig.tight_layout()
plt.legend()


plt.savefig("visuals.png")


plt.show()


