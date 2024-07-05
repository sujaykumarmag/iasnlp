

from src.dataset import TranslationDataset
from src.utils import get_exp

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter

import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml

import evaluate 
sacrebleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

ter_metric = evaluate.load("ter")



class NormalTrain():
    
    def __init__(self,args, dataset, model, tokenizer):
        self.args = args
        self.batchsize = args.batchsize
        self.lr = args.lr
        self.numepochs = args.numepochs
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
    
        train_loader, val_loader, test_loader = self.get_dataloaders()
        self.train(train_loader,val_loader,test_loader)
        
        
        
    def get_dataloaders(self):
        df = self.dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)[:20000]
        print(df)
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
        
        train_dataset = TranslationDataset(train_df, self.tokenizer)
        val_dataset = TranslationDataset(val_df, self.tokenizer)
        test_dataset = TranslationDataset(test_df, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        return train_loader, val_loader, test_loader
    
    
    def train_epoch_enc_dec(self,model, data_loader, optimizer, scheduler, device, epoch):
        model.train()
        total_loss = 0
        for batch in data_loader:
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
    
    
    def train_epoch_dec_only(self,model, data_loader, optimizer, scheduler, device, epoch):
        model.train()
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = self.tokenizer(batch['source_text'], return_tensors='pt', padding=True).input_ids.squeeze()
            labels = self.tokenizer(batch['target_text'], return_tensors='pt', padding=True).input_ids.squeeze()
        
            input_ids = batch['source_ids'].to(device)
            attention_mask = batch['source_mask'].to(device)
            labels = batch['target_ids'].to(device)
            
        
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
        
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(data_loader)
        return avg_train_loss
    
    
    def eval_epoch_dec_only(self, model,val_dataloader,device, epoch):
        # Validation phase
        model.eval()
        comet_scores = []
        chrf_scores = []
        bleu_scores = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = self.tokenizer(batch['source_text'], return_tensors='pt', padding=True).input_ids
                labels = self.tokenizer(batch['target_text'], return_tensors='pt', padding=True).input_ids
                input_ids = batch['source_ids'].to(device)
                attention_mask = batch['source_mask'].to(device)
                labels = batch['target_ids'].to(device)
                
                
                
                
                outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
                
                predictions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
                references = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
                
                bleu_score = self.compute_bleu_score(predictions, references)
                ter_score = self.compute_ter_score(predictions, references)
                chrf_score = self.compute_chrf_score(predictions, references)
                
                comet_scores.append(ter_score)
                bleu_scores.append(bleu_score)
                chrf_scores.append(chrf_score)
    
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        avg_ter_score = sum(comet_scores) / len(comet_scores)
        avg_chrf_score = sum(chrf_scores) / len(chrf_scores)
            
    
        return avg_bleu_score, avg_ter_score, avg_chrf_score
    
    
    def compute_bleu_score(self,predictions, references):
        predictions_list = [str(pred) for pred in predictions]
        references_list = [[str(ref)] for ref in references]
        # bleu = sacrebleu.corpus_bleu(predictions_list, references_list)
        # preds [p1,p2,p3] references [[r1],[r2],[r3]]
        bleu = sacrebleu_metric.compute(predictions= predictions_list, references = references_list)
        return bleu["score"]
    
    def compute_chrf_score(self,predictions, references):
        predictions_list = [str(pred) for pred in predictions]
        references_list = [[str(ref)] for ref in references]
        # score = sacrebleu.corpus_chrf(predictions_list, references_list)
        chrf = chrf_metric.compute(predictions= predictions_list, references = references_list)
        return chrf["score"]
    
    def compute_ter_score(self,predictions, references):
        predictions_list = [str(pred) for pred in predictions]
        references_list = [[str(ref) ]for ref in references]
        # ter = sacrebleu.corpus_ter(predictions_list, references_list)
        ter_score = ter_metric.compute(predictions= predictions_list, references = references_list)
        return ter_score["score"]


    
    def test_epoch_dec_only(self, model,val_dataloader,device, epoch):
        # Validation phase
        model.eval()
        comet_scores = []
        chrf_scores = []
        bleu_scores = []
        generated = []
        org = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = self.tokenizer(batch['source_text'], return_tensors='pt', padding=True).input_ids
                labels = self.tokenizer(batch['target_text'], return_tensors='pt', padding=True).input_ids
                
                input_ids = batch['source_ids'].to(device)
                attention_mask = batch['source_mask'].to(device)
                labels = batch['target_ids'].to(device)
                
                
                outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
                predictions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
                references = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
                print(predictions,outputs)
                bleu_score = self.compute_bleu_score(predictions, references)
                ter_score = self.compute_ter_score(predictions, references)
                chrf_score = self.compute_chrf_score(predictions, references)
                
                comet_scores.append(ter_score)
                bleu_scores.append(bleu_score)
                chrf_scores.append(chrf_score)
                # print(batch['source_text'])
                # tokenizer.decode(dec_only_model.generate(inp, max_length=50, num_return_sequences=1)[0], skip_special_tokens=True)
                
                # # predictions = self.tokenizer.decode(self.model.generate(input_ids, max_length=50, num_return_sequences=1)[0], skip_special_tokens=True)
                # predictions = self.tokenizer.decode(self.model.generate(input_ids, use_cache=True, 
                #            num_beams=4, max_length=150, min_length=1, early_stopping=True, 
                #            decoder_start_token_id=self.tokenizer._convert_token_to_id_with_added_voc("<2en>"))[0], skip_special_tokens=True)
                # print(predictions)
                
                # references = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
                generated.append(predictions)
                org.append(references)
            
    
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        avg_ter_score = sum(comet_scores) / len(comet_scores)
        avg_chrf_score = sum(chrf_scores) / len(chrf_scores)
        data = {
            'Generated Text': generated,
            'Original Text': org
        }
        df = pd.DataFrame(data)
    
        return df, avg_bleu_score, avg_ter_score, avg_chrf_score
        
    def train(self, train_loader, val_loader, test_loader):
        output_dir = get_exp()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(train_loader) * self.numepochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        print(self.model)
        metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'TER', 'chrf','bleu'])
        
        for epoch in tqdm(range(self.numepochs),desc=f"Training and Validation Epochs"):
            train_loss = self.train_epoch_dec_only(self.model, train_loader, optimizer, scheduler, device, epoch)
            bleu, ter, chrf = self.eval_epoch_dec_only(self.model, val_loader, device, epoch)
            tqdm.write(f"Epoch {epoch + 1},  Training loss: {train_loss}")
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'TER': ter,
                'chrf': chrf,
                'bleu': bleu,
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)
            
        fig, ax1 = plt.subplots(figsize=(12, 8))
        # Define the x-axis (epochs)
        epochs = metrics_df['epoch']
        # Plot training loss
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:red')
        ax1.plot(epochs, metrics_df['train_loss'], label='Train Loss', color='tab:red', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Scores', color='tab:blue')
        ax2.plot(epochs, metrics_df['bleu'], label='BLEU Score', color='tab:blue', marker='x')
        ax2.plot(epochs, metrics_df['TER'], label='TER Score', color='tab:orange', marker='s')
        ax2.plot(epochs, metrics_df['chrf'], label='CHRF Score', color='tab:green', marker='^')
        fig.tight_layout()
        plt.legend()
        plt.savefig(f"{output_dir}/visuals.png")


        test_df, bleu, ter, chrf = self.test_epoch_dec_only(self.model,test_loader,device, 0)
        epoch_metrics = {
                'epoch': 'Test',
                'train_loss': 'Test',
                'TER': ter,
                'chrf': chrf,
                'bleu': bleu,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)
        metrics_df.to_csv(f"{output_dir}/results.csv",index=False)
        test_df.to_csv(f"{output_dir}/test_outputs.csv",index=False)
        args_dict = vars(self.args)
        with open(output_dir+"/args.yaml", 'w') as file:
            yaml.dump(args_dict, file, default_flow_style=False)
            
        torch.save(self.model.state_dict(), f"{output_dir}/best.pt")
        
        print(metrics_df)
        print(test_df)

        
        
        
        
  
    
    
    
        
        
        
    
    
    
    