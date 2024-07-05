import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.xlnet.modeling_xlnet import XLNetLMHeadModel
from src.xlnet.configuration_xlnet import XLNetConfig



import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from src.mbart.configuration_mbart import MBartConfig
from src.mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
from src.mbart.tokenization_mbart import MBartTokenizer


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
data_lang1['x'] = data_lang1['en'] + ' </s>' + '<2hi>'
data_lang1['y'] = '<2hi> ' + data_lang1['hi'] + ' </s>'

df = data_lang1[:20]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

























# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

# Token IDs
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")


# Define a PyTorch Dataset
class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data.iloc[idx]['x']
        output_ids = self.data.iloc[idx]['y']
        
        
        return {'input_ids': input_ids, 'output_ids': output_ids}

# Create an instance of the Dataset
dataset = TranslationDataset(df)

# Create a PyTorch DataLoader
batch_size = 2  # Set your desired batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)





# Encoder-Decoder Model Configuration
enc_dec_config = MBartConfig(
    vocab_size=64014,
    bos_token_id=64000,
    activation_dropout=0.1,
    attention_dropout=0.1,
    encoder_layers=2,
    decoder_layers=3,
    pad_token_id=0,
    eos_token_id=64001
)
enc_dec_model = MBartForConditionalGeneration(config=enc_dec_config)

# Decoder-Only Model Configuration
dec_only_config = XLNetConfig(
    vocab_size=64014,
    bos_token_id=64000,
    n_layer=6,
    pad_token_id=0,
    eos_token_id=64001
)
dec_only_model = XLNetLMHeadModel(config=dec_only_config)











# Training settings
epochs = 2
learning_rate = 1e-4

# Define optimizers and loss function
enc_dec_optimizer = torch.optim.Adam(enc_dec_model.parameters(), lr=learning_rate)
dec_only_optimizer = torch.optim.Adam(dec_only_model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()



# # Training loop for Encoder-Decoder model
# for epoch in range(epochs):
#     enc_dec_model.train()
#     total_loss_enc_dec = 0.0
#     for batch in data_loader:
#         input_ids = batch['input_ids']
        
#         output_ids = batch['output_ids']
        
#         input_ids = tokenizer(input_ids, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
#         output_ids = tokenizer(output_ids, add_special_tokens=False, return_tensors="pt", padding=True).input_ids

        
#         enc_dec_optimizer.zero_grad()
#         output_enc_dec = enc_dec_model(input_ids=input_ids,  labels=output_ids)
#         loss_enc_dec = output_enc_dec.loss
#         loss_enc_dec.backward()
#         enc_dec_optimizer.step()
        
#         total_loss_enc_dec += loss_enc_dec.item()
    
#     avg_loss_enc_dec = total_loss_enc_dec / len(data_loader)
#     print(f"Epoch [{epoch + 1}/{epochs}], Encoder-Decoder Model Loss: {avg_loss_enc_dec}")

# print("Encoder-Decoder Model Training finished.")





for epoch in range(epochs):
    dec_only_model.train()
    total_loss_dec_only = 0.0
    for batch in data_loader:
        
        input_ids = batch['input_ids']
        output_ids = batch['output_ids']
        print(batch)
        
        input_ids = tokenizer(input_ids, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        output_ids = tokenizer(output_ids, add_special_tokens=False, return_tensors="pt", padding=True).input_ids[:, 1:].contiguous()
        
        # Print shapes for debugging
        print(input_ids)
        
        dec_only_optimizer.zero_grad()
        output_dec_only = dec_only_model(input_ids=input_ids, labels=output_ids)
        loss_dec_only = output_dec_only.loss
        loss_dec_only.backward()
        dec_only_optimizer.step()
        
        total_loss_dec_only += loss_dec_only.item()
    
    avg_loss_dec_only = total_loss_dec_only / len(data_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Decoder-Only Model Loss: {avg_loss_dec_only}")

print("Decoder-Only Model Training finished.")



# Print number of trainable parameters
print("Encoder-Decoder Model Trainable Parameters:", enc_dec_model.num_parameters())
print("Decoder-Only Model Trainable Parameters:", dec_only_model.num_parameters())
