########################################################################################################################################
# Author : SujayKumar Reddy M
# Date Created : 2 July 2024
# Project : Decoder and Encoder-Decoder model Comparison with the ROC and Benchmarks
# File Usage : For conditional and total training all the 3 major Experiments
########################################################################################################################################


import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import pandas as pd
import numpy as np
from src.icl import InContextLearning
from src.baselines import BaselineLearning

parser = argparse.ArgumentParser(description="This is an  Argument Parser for the Encoder-Decoder Comparison")

parser.add_argument('experiment',type=str,metavar="The Major Experiment type (icl, finetune, baseline)")


# Hyper parameters for ICL Experiments paradigm
parser.add_argument("--retriever", type=str, default="random")
parser.add_argument("--prompt_template", type=str, default="</E></X>=</Y>")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--ice_num", type=int, default=8)
parser.add_argument("--oracle", default=False, action="store_true")
parser.add_argument("--disorder", default=False, action="store_true")
parser.add_argument("--repeat", default=False, action="store_true")
parser.add_argument("--reverse_direction", default=False, action="store_true")
parser.add_argument("--direction_order", nargs="+", type=int, default=None)
parser.add_argument("--cross_lang", default=False, action="store_true")
parser.add_argument("--ex_lang", nargs="+", type=str, default=None)
parser.add_argument("--lang_order", nargs="+", type=int, default=None)
parser.add_argument('--run_all_icl',type=bool, default=False)


# Hyperparameters for Finetuning LORA
parser.add_argument("--lora_alpha",type=float,default=16)
parser.add_argument("--lora_dropout",default=0.05,type=float)




# Common parameters for all three Experiment Paradigms
parser.add_argument("--lang_pair", type=str, default="eng-hin")
parser.add_argument("--model_name", type=str, default="google/mt5-base")
parser.add_argument("--tokenizer_name", type=str, default="google/mt5-base")




# Common parameters for finetuning and baselines Experiment Paradigms
parser.add_argument("--multi", nargs="+", type=str, default='eng hin mar')
parser.add_argument("--model_type",type=str,default="enc_dec")
parser.add_argument("--one2many",type=bool,default=False)
parser.add_argument("--many2one",type=bool, default=False)
parser.add_argument("--numepochs",type=int,default=5)
parser.add_argument("--batchsize",type=int,default=10)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument("--output_dir", type=str, default="runs/")


args = parser.parse_args()
print(args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(42)

dataset_dir = "./datasets/wiki/"
if args.experiment == "icl":
    paradigm = InContextLearning(args,dataset_dir=dataset_dir)
    paradigm.train()
elif args.experiment == "baseline":
    paradigm = BaselineLearning(args,dataset_dir=dataset_dir)