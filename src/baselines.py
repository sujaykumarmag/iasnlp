


import pandas as pd
import numpy as np
import os

from src.utils import get_contents, get_duals,remove_file_artifacts
from src.training.normal_train import NormalTrain

from sklearn.model_selection import train_test_split
from transformers import  AutoTokenizer
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.xlnet.modeling_xlnet import XLNetLMHeadModel
from src.xlnet.configuration_xlnet import XLNetConfig
from src.decoder_only import DecoderOnlyTransformer, initialize_weights


from src.mbart.configuration_mbart import MBartConfig
from src.mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
from src.mbart.tokenization_mbart import MBartTokenizer


class BaselineLearning():
    
    def __init__(self,args,dataset_dir):
        self.args = args
        self.dataset_dir = dataset_dir
        self.datatypes = os.listdir(dataset_dir)
        self.datatypes = remove_file_artifacts(self.datatypes)
        self.datatypes.remove("icl")
        self.model_type = args.model_type
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", 
                                                       do_lower_case=False, use_fast=False, keep_accents=True,padding_side='left')
        bos_id = self.tokenizer._convert_token_to_id_with_added_voc("<s>")
        eos_id = self.tokenizer._convert_token_to_id_with_added_voc("</s>")
        pad_id = self.tokenizer._convert_token_to_id_with_added_voc("<pad>")
        if self.model_type == "enc_dec":
            enc_dec_config = MBartConfig(vocab_size=64014, bos_token_id=64000, activation_dropout=0.1, attention_dropout=0.1,
                             encoder_layers=2, decoder_layers=3, pad_token_id=0, eos_token_id=64001)
            self.model = MBartForConditionalGeneration(config=enc_dec_config)
        else:
            from src.xlnet.tokenization_xlnet import XLNetTokenizer
            self.tokenizer = XLNetTokenizer(self.tokenizer.vocab_file)
            dec_only_config = XLNetConfig(vocab_size = 64014, bos_token_id= 64000, n_layer=6,pad_token_id=0,eos_token_id=64001)
            self.model = XLNetLMHeadModel(config=dec_only_config)
            
            
        self.one2many = args.one2many
        if self.one2many:
            self.dataset = self.prepare_dataset_multi()
        elif self.args.many2one:
            self.dataset = self.prepare_dataset_multi()
            temp = self.dataset["y"]
            self.dataset["y"] = self.dataset["x"]
            self.dataset["x"] = temp
        else:
            self.dataset = self.prepare_dataset()
        
        trainer = NormalTrain(args, self.dataset, self.model, self.tokenizer)
        
            
    def prepare_dataset(self):
        src, tgt  =  self.args.lang_pair.split('-')
        LANG1 = None
        if src == "eng":
            LANG1 = tgt
        else:
            LANG1 = src
        data_lang1 = get_duals(datatypes=self.datatypes,lang=LANG1,PATH=self.dataset_dir)
        data_lang1 = pd.DataFrame(data_lang1)
        
        data_lang1['x'] = data_lang1['en'] + '</s>' +'<2en>'
        data_lang1['y'] = '<2hi>' + data_lang1['hin'] + '</s>'
        return data_lang1[['x', 'y']]
        
            
            
    def prepare_dataset_multi(self):
        self.args.multi = self.args.multi.split(" ")
        LANG1 = self.args.multi[1]
        LANG2 = self.args.multi[2]
        print(LANG1,LANG2,self.datatypes)
        data_lang1 = get_duals(datatypes=self.datatypes,lang=LANG1,PATH=self.dataset_dir)
        data_lang2 = get_duals(datatypes=self.datatypes,lang=LANG2,PATH=self.dataset_dir)
        
        data_lang1 = pd.DataFrame(data_lang1)
        data_lang2 = pd.DataFrame(data_lang2)
        
        # Annotate the DataFrame
        data_lang1['en'] = data_lang1['en'] + ' </s> <2en>'
        data_lang1[LANG1] = '<2hi> ' + data_lang1['hin'] + ' </s>'
        
        data_lang2['en'] = data_lang2['en'] + '</s>' +'<2en>'
        data_lang2[LANG2] = '<2mar>' + data_lang2['mar'] + '</s>'
        
        combined = pd.concat([data_lang1[['en', LANG1]], data_lang2[['en', LANG2]]], ignore_index=True)
        combined["y"] = combined[LANG1].fillna('') + combined[LANG2].fillna('')
        combined = combined.drop([LANG1,LANG2],axis=1)
        combined["x"] = combined["en"]
        return combined[['x','y']]
    





        
        