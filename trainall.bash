#!/bin/bash

python3 train.py icl --lang_pair eng-hin --model_name facebook/xglm-564M
python3 train.py icl --lang_pair eng-hin --run_all_icl true --model_name facebook/xglm-564M
python3 train.py icl --lang_pair eng-hin --run_all_icl true --model_name google/mt5-base

python3 train.py baseline --lang_pair eng-hin --model_type enc_dec 
python3 train.py baseline --lang_pair eng-hin --model_type enc_dec --one2many True
python3 train.py baseline --lang_pair eng-hin --model_type enc_dec --many2one True
python3 train.py baseline --lang_pair eng-hin --model_type dec_only 
python3 train.py baseline --lang_pair eng-hin --model_type dec_only --one2many True
python3 train.py baseline --lang_pair eng-hin --model_type dec_only --many2one True