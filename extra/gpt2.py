
from transformers import AutoTokenizer
import torch
from src.gpt2 import GPT2Config, GPT2LMHeadModel
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



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
data_lang1['x'] = data_lang1['en'] + '</s>' +'<2hi>'
data_lang1['y'] = '<2hi>' + data_lang1['hi'] + '</s>'


tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)


df = data_lang1[:20]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

# Decoder-Only Model Configuration
dec_only_config = GPT2Config(
    vocab_size=64014,
    bos_token_id=64000,
    n_layer=6,
    pad_token_id=0,
    eos_token_id=64001
)
model = GPT2LMHeadModel(config=dec_only_config)

print("Decoder-Only Model Trainable Parameters:", model.num_parameters())


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
        
        

train_dataset = TranslationDataset(train_df,tokenizer)
val_dataset = TranslationDataset(val_df,tokenizer)
test_dataset = TranslationDataset(test_df,tokenizer)

# Example dataloader initialization
batch_size = 16  # Adjust batch size as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





# # Tokenize input and output sequences
# inputs = tokenizer(x, return_tensors="pt", padding=True)
# outputs = tokenizer(y, return_tensors="pt", padding=True)

# # Extract input and output tensors
# input_ids = inputs.input_ids
# labels = outputs.input_ids

# # Define training parameters
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 3
criterion = torch.nn.CrossEntropyLoss()
import sacrebleu
    

# #  Training function
# def train(model, dataloader, optimizer, num_epochs=3):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for batch in dataloader:
#             input_ids = tokenizer(batch['source_text'], return_tensors='pt', padding=True).input_ids
#             labels = tokenizer(batch['target_text'], return_tensors='pt', padding=True).input_ids
            
#             optimizer.zero_grad()
            
#             outputs = model(input_ids=input_ids, labels=labels)
#             loss = outputs.loss
            
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         epoch_loss = running_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_dataloader:
#                 input_ids = tokenizer(batch['source_text'], return_tensors='pt', padding=True).input_ids
#                 labels = tokenizer(batch['target_text'], return_tensors='pt', padding=True).input_ids
                
#                 outputs = model(input_ids=input_ids, labels=labels)
#                 loss = outputs.loss
                
#                 val_loss += loss.item()
        
#         val_loss /= len(val_dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")


# # Example: Train using the train DataLoader
# train(model, train_dataloader, optimizer)


# # # Example of using the model for translation (inference)
# # translated_ids = model.generate(input_ids=input_ids, max_length=100, num_beams=5, early_stopping=True)
# # translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

# # print("Translated output:", translated_text)



# Evaluation functions
def compute_bleu_score(predictions, references):
    predictions_list = [str(pred) for pred in predictions]
    references_list = [str(ref) for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions_list, [references_list])
    return bleu.score

def compute_chrf_score(predictions, references):
    predictions_list = [str(pred) for pred in predictions]
    references_list = [str(ref) for ref in references]
    score = sacrebleu.corpus_chrf(predictions_list, [references_list])
    return score.score

def compute_ter_score(predictions, references):
    predictions_list = [str(pred) for pred in predictions]
    references_list = [str(ref) for ref in references]
    ter = sacrebleu.corpus_ter(predictions_list, [references_list])
    return ter.score









# Training function with evaluation metrics
def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=3):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            input_ids = tokenizer(batch['source_text'], return_tensors='pt', padding=True).input_ids
            labels = tokenizer(batch['target_text'], return_tensors='pt', padding=True).input_ids
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        
        # Validation phase
        model.eval()
        comet_scores = []
        chrf_scores = []
        bleu_scores = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = tokenizer(batch['source_text'], return_tensors='pt', padding=True).input_ids
                labels = tokenizer(batch['target_text'], return_tensors='pt', padding=True).input_ids
                
                outputs = model.generate(input_ids=input_ids, max_length=50, num_beams=5, early_stopping=True)
                predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
                references = batch['target_text']
                bleu_score = compute_bleu_score(predictions, references)
                ter_score = compute_ter_score(predictions, references)
                chrf_score = compute_chrf_score(predictions, references)
                comet_scores.append(ter_score)
                bleu_scores.append(bleu_score)
                chrf_scores.append(chrf_score)
    
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        avg_ter_score = sum(comet_scores) / len(comet_scores)
        avg_chrf_score = sum(chrf_scores) / len(chrf_scores)
            
    
        print(avg_bleu_score, avg_ter_score, avg_chrf_score)
                
        
    
        # # Compute evaluation metrics
        # references = val_df['target_text'].tolist()
        # bleu_score = compute_bleu_score(val_predictions, references)
        # chrf_score = compute_chrf_score(val_predictions, references)
        # ter_score = compute_ter_score(val_predictions, references)
        
        # print(f"Epoch [{epoch+1}/{num_epochs}], BLEU: {bleu_score:.4f}, CHRF: {chrf_score:.4f}, TER: {ter_score:.4f}")

# Example: Train using the train DataLoader and evaluate on the val DataLoader
train(model, train_dataloader, val_dataloader, optimizer, criterion)

