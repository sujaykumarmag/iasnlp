########################################################################################################################################
# Author : SujayKumar Reddy M
# Date Created : 2 July 2024
# Project : Decoder and Encoder-Decoder model Comparison with the ROC and Benchmarks
# File Usage : For In-Context-Learning Paradigm
# References : 
#       * load_dataset, get_rtr and test functions are taken from https://github.com/NJUNLP/MMT-LLM
########################################################################################################################################


# Imports
import os
import sys
from os.path import dirname as d
from os.path import abspath, join
root = d(d(abspath(__file__)))
sys.path.append(root)

import argparse
from typing import List, Union, Optional
from copy import deepcopy
import numpy as np
import itertools
import random
import pandas as pd
import yaml

from src.openicl.icl_dataset_reader import IclDatasetReader
from src.openicl.icl_retriever.icl_base_retriever import IclBaseRetriever
from src.openicl.icl_retriever.icl_bm25_retriever import IclBM25Retriever 
from src.openicl.icl_retriever.icl_topk_mdl_retriever import IclTopkMDLRetriever
from src.openicl.icl_retriever.icl_random_retriever import IclRandomRetriever 
from src.openicl.icl_retriever.icl_topk_retriever import IclTopkRetriever
from src.openicl.icl_retriever.icl_votek_retriever import IclVotekRetriever 
from src.openicl.icl_prompt_template import IclPromptTemplate
from src.openicl.icl_inferencer.icl_gen_inferencer import IclGenInferencer
from datasets import load_dataset, Dataset, DatasetDict


from src.utils import remove_file_artifacts, get_exp


# Configurations for ICL
lang_config = ['afr', 'amh', 'ara', 'hye', 'asm', 'ast', 'azj', 'bel', 'ben', 'bos', 'bul', 'mya', 'cat', 'ceb', 'zho_simpl', 'zho_trad', 'hrv', 'ces', 'dan', 'nld', 'eng', 'est', 'tgl', 'fin', 'fra', 'ful', 'glg', 'lug', 'kat', 'deu', 'ell', 'guj', 'hau', 'heb', 'hin', 'hun', 'isl', 'ibo', 'ind', 'gle', 'ita', 'jpn', 'jav', 'kea', 'kam', 'kan', 'kaz', 'khm', 'kor', 'kir', 'lao', 'lav', 'lin', 'lit', 'luo', 'ltz', 'mkd', 'msa', 'mal', 'mlt', 'mri', 'mar', 'mon', 'npi', 'nso', 'nob', 'nya', 'oci', 'ory', 'orm', 'pus', 'fas', 'pol', 'por', 'pan', 'ron', 'rus', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'ckb', 'spa', 'swh', 'swe', 'tgk', 'tam', 'tel', 'tha', 'tur', 'ukr', 'umb', 'urd', 'uzb', 'vie', 'cym', 'wol', 'xho', 'yor', 'zul']
ISO_639_2_T_abbr2full = {
    "afr": "Afrikaans", "amh": "Amharic", "ara": "Arabic", "hye": "Armenian", "asm": "Assamese", "ast": "Asturian", "azj": "Azerbaijani", "bel": "Belarusian", "ben": "Bengali", "bos": "Bosnian", "bul": "Bulgarian", "mya": "Burmese", "cat": "Catalan", "ceb": "Cebuano", "zho_simpl": "Chinese Simpl", "zho_trad": "Chinese Trad", "hrv": "Croatian", "ces": "Czech", "dan": "Danish", "nld": "Dutch", "eng": "English", "est": "Estonian", "tgl": "Filipino", "fin": "Finnish", "fra": "French", "ful": "Fulah", "glg": "Galician", "lug": "Ganda", "kat": "Georgian", "deu": "German", "ell": "Greek", "guj": "Gujarati", "hau": "Hausa", "heb": "Hebrew", "hin": "Hindi", "hun": "Hungarian", "isl": "Icelandic", "ibo": "Igbo", "ind": "Indonesian", "gle": "Irish", "ita": "Italian", "jpn": "Japanese", "jav": "Javanese", "kea": "Kabuverdianu", "kam": "Kamba", "kan": "Kannada", "kaz": "Kazakh", "khm": "Khmer", "kor": "Korean", "kir": "Kyrgyz", "lao": "Lao", "lav": "Latvian", "lin": "Lingala", "lit": "Lithuanian", "luo": "Luo", "ltz": "Luxembourgish", "mkd": "Macedonian", "msa": "Malay", "mal": "Malayalam", "mlt": "Maltese", "mri": "Maori", "mar": "Marathi", "mon": "Mongolian", "npi": "Nepali", "nso": "Northern Sotho", "nob": "Norwegian", "nya": "Nyanja", "oci": "Occitan", "ory": "Oriya", "orm": "Oromo", "pus": "Pashto", "fas": "Persian", "pol": "Polish", "por": "Portuguese", "pan": "Punjabi", "ron": "Romanian", "rus": "Russian", "srp": "Serbian", "sna": "Shona", "snd": "Sindhi", "slk": "Slovak", "slv": "Slovenian", "som": "Somali", "ckb": "Sorani Kurdish", "spa": "Spanish", "swh": "Swahili", "swe": "Swedish", "tgk": "Tajik", "tam": "Tamil", "tel": "Telugu", "tha": "Thai", "tur": "Turkish", "ukr": "Ukrainian", "umb": "Umbundu", "urd": "Urdu", "uzb": "Uzbek", "vie": "Vietnamese", "cym": "Welsh", "wol": "Wolof", "xho": "Xhosa", "yor": "Yoruba", "zul": "Zulu"
}
ISO_639_1_abbr2full = {
    'ab': 'Abkhaz', 'aa': 'Afar', 'af': 'Afrikaans', 'ak': 'Akan', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic', 'an': 'Aragonese', 'hy': 'Armenian', 'as': 'Assamese', 'av': 'Avaric', 'ae': 'Avestan', 'ay': 'Aymara', 'az': 'Azerbaijani', 'bm': 'Bambara', 'ba': 'Bashkir', 'eu': 'Basque', 'be': 'Belarusian', 'bn': 'Bengali', 'bh': 'Bihari', 'bi': 'Bislama', 'bs': 'Bosnian', 'br': 'Breton', 'bg': 'Bulgarian', 'my': 'Burmese', 'ca': 'Catalan', 'ch': 'Chamorro', 'ce': 'Chechen', 'ny': 'Nyanja', 'zh': 'Chinese', 'cv': 'Chuvash', 'kw': 'Cornish', 'co': 'Corsican', 'cr': 'Cree', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'dv': 'Divehi; Maldivian;', 'nl': 'Dutch', 'dz': 'Dzongkha', 'en': 'English', 'eo': 'Esperanto', 'et': 'Estonian', 'ee': 'Ewe', 'fo': 'Faroese', 'fj': 'Fijian', 'fi': 'Finnish', 'fr': 'French', 'ff': 'Fula', 'gl': 'Galician', 'ka': 'Georgian', 'de': 'German', 'el': 'Greek', 'gn': 'Guaraní', 'gu': 'Gujarati', 'ht': 'Haitian', 'ha': 'Hausa', 'he': 'Hebrew', 'hz': 'Herero', 'hi': 'Hindi', 'ho': 'Hiri Motu', 'hu': 'Hungarian', 'ia': 'Interlingua', 'id': 'Indonesian', 'ie': 'Interlingue', 'ga': 'Irish', 'ig': 'Igbo', 'ik': 'Inupiaq', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese', 'jv': 'Javanese', 'kl': 'Kalaallisut', 'kn': 'Kannada', 'kr': 'Kanuri', 'ks': 'Kashmiri', 'kk': 'Kazakh', 'km': 'Khmer', 'ki': 'Kikuyu, Gikuyu', 'rw': 'Kinyarwanda', 'ky': 'Kyrgyz', 'kv': 'Komi', 'kg': 'Kongo', 'ko': 'Korean', 'ku': 'Kurdish', 'kj': 'Kwanyama, Kuanyama', 'la': 'Latin', 'lb': 'Luxembourgish', 'lg': 'Luganda', 'li': 'Limburgish', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'lu': 'Luba-Katanga', 'lv': 'Latvian', 'gv': 'Manx', 'mk': 'Macedonian', 'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese', 'mi': 'Māori', 'mr': 'Marathi', 'mh': 'Marshallese', 'mn': 'Mongolian', 'na': 'Nauru', 'nv': 'Navajo, Navaho', 'nb': 'Norwegian Bokmål', 'nd': 'North Ndebele', 'ne': 'Nepali', 'ng': 'Ndonga', 'nn': 'Norwegian Nynorsk', 'no': 'Norwegian', 'ii': 'Nuosu', 'nr': 'South Ndebele', 'oc': 'Occitan', 'oj': 'Ojibwe, Ojibwa', 'cu': 'Old Church Slavonic', 'om': 'Oromo', 'or': 'Oriya', 'os': 'Ossetian, Ossetic', 'pa': 'Punjabi', 'pi': 'Pāli', 'fa': 'Persian', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'qu': 'Quechua', 'rm': 'Romansh', 'rn': 'Kirundi', 'ro': 'Romanian', 'ru': 'Russian', 'sa': 'Sanskrit Saṁskṛta)', 'sc': 'Sardinian', 'sd': 'Sindhi', 'se': 'Northern Sami', 'sm': 'Samoan', 'sg': 'Sango', 'sr': 'Serbian', 'gd': 'Scottish Gaelic', 'sn': 'Shona', 'si': 'Sinhala, Sinhalese', 'sk': 'Slovak', 'sl': 'Slovene', 'so': 'Somali', 'st': 'Southern Sotho', 'es': 'Spanish', 'su': 'Sundanese', 'sw': 'Swahili', 'ss': 'Swati', 'sv': 'Swedish', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'bo': 'Tibetan', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tn': 'Tswana', 'to': 'Tonga', 'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian', 'ug': 'Uighur, Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vi': 'Vietnamese', 'vo': 'Volapük', 'wa': 'Walloon', 'cy': 'Welsh', 'wo': 'Wolof', 'fy': 'Western Frisian', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang, Chuang', 'zu': 'Zulu'
}


class InContextLearning():
    
    def __init__(self,args,dataset_dir):
        self.args = args
        self.all_pairs = ['eng-hin','hin-eng','eng-tel','tel-eng','eng-tam','tam-eng','eng-mal','mal-eng']
        self.dataset_dir = os.path.join(dataset_dir,"icl")
    
    
    def load_dataset(self,src_lang, tgt_lang, args, pair=None):
        if src_lang=="eng":
            pair = tgt_lang
        else:
            pair = src_lang
            
        valid_src = [line.strip() for line in open(join(self.dataset_dir, 'eng-'+pair+'/{}.dev'.format(src_lang))).readlines()]
        valid_tgt = [line.strip() for line in open(join(self.dataset_dir, 'eng-'+pair+'/{}.dev'.format(tgt_lang))).readlines()]
        
        test_src = [line.strip() for line in open(join(self.dataset_dir, 'eng-'+pair+'/{}.devtest'.format(src_lang))).readlines()]
        test_tgt = [line.strip() for line in open(join(self.dataset_dir, 'eng-'+pair+'/{}.devtest'.format(tgt_lang))).readlines()]
        
        valid_dataset = Dataset.from_dict({'translation': [{src_lang: src, tgt_lang: tgt} for src, tgt in zip(valid_src, valid_tgt)], 
                                       src_lang: valid_src, 
                                       tgt_lang: valid_tgt})
        test_dataset = Dataset.from_dict({'translation': [{src_lang: src, tgt_lang: tgt} for src, tgt in zip(test_src, test_tgt)],
                                      src_lang: test_src,
                                      tgt_lang: test_tgt})
        return DatasetDict({'dev': valid_dataset, 'devtest': test_dataset})
    
    
    def get_rtr(self,src, tgt, args):
        
        dr = IclDatasetReader(self.load_dataset(src, tgt, args), ctx_list=[src], pred_label=tgt)
        tp_str = args.prompt_template
        
        if "[source]" in tp_str:
            tp_str = tp_str.replace("[source]", src)
            
        if "[target]" in tp_str:
            tp_str = tp_str.replace("[target]", tgt)
            
        tp = IclPromptTemplate(
            tp_str, 
            ctx_token_dict={src: "</X>", tgt: "</Y>"},
            ice_token="</E>"
            )
        if args.retriever == "random":
            rtr = IclRandomRetriever(dr, ice_template=tp, ice_num=args.ice_num, select_split="dev", generation_split="devtest", seed=args.seed)
        elif args.retriever == "bm25":
            rtr = IclBM25Retriever(dr, ice_template=tp, ice_num=args.ice_num, select_split="dev", generation_split="devtest", oracle=args.oracle)
        elif args.retriever == "topk":
            rtr = IclTopkRetriever(dr, ice_template=tp, ice_num=args.ice_num, batch_size=16, select_split="dev", generation_split="devtest")
        elif args.retriever == "topk_mdl":
            rtr = IclTopkMDLRetriever(dr, ice_template=tp, ice_num=args.ice_num, batch_size=16, select_split="dev", generation_split="devtest")
        elif args.retriever == "votek":
            rtr = IclVotekRetriever(dr, ice_template=tp, ice_num=args.ice_num, batch_size=16, select_split="dev", generation_split="devtest")
        else:
            raise NotImplementedError
        return rtr
    
    
        
    def test(self, args, lang_pair, metric,output_dir):
        src, tgt = lang_pair.split('-')
        
        if args.reverse_direction:
            rtrs = [self.get_rtr(src, tgt, args), self.get_rtr(tgt, src, args)]
            rtr_order = args.direction_order
        elif args.cross_lang:
            rtrs = [self.get_rtr(src, tgt, args)]
            for lang in args.ex_lang:
                rtrs.append(self.get_rtr(lang, tgt, args))
            rtr_order = args.lang_order
        else:
            rtrs = [self.get_rtr(src, tgt, args)]
            rtr_order = None
            
        infr = IclGenInferencer(
            rtrs, metric=metric, max_model_token_num=1800, batch_size=args.batchsize, rtr_order=rtr_order,
            ice_operator=None,
            model_name=args.model_name, tokenizer_name=args.tokenizer_name,
            output_json_filepath=output_dir
            )
        # try:
        score = infr.score(src_lang=src,tgt_lang=tgt,output_dir=output_dir)
        # except:
            # print("Error while Calculation of score")
            # score = 0.0
        print(metric,score)
        print("inference finished...")
        return score
    
    
    def train(self):
        self.lang_pair = self.args.lang_pair
        output_dir = get_exp()
        if self.args.run_all_icl:
            self.train_all(output_dir)
            return
        metrics = ['bleu','squad']
        res = []
        fullscores = []
        for i in metrics:
            score = self.test(self.args,self.lang_pair,metric=i,output_dir=output_dir)
            fullscores.append(np.float32(score))
        
        print(fullscores)
        res.append({
            "lang_pair": self.lang_pair,
            "bleu": fullscores[0],
            "squad": fullscores[1]
        })
        results_df = pd.DataFrame(res)
        results_df.to_csv(os.path.join(output_dir,"results.csv"),index=False)
        args_dict = vars(self.args)
        with open(output_dir+"/args.yaml", 'w') as file:
            yaml.dump(args_dict, file, default_flow_style=False)
        
        
        
        
    def train_all(self,output_dir):
        res = []
        for i in self.all_pairs:
            metrics = ['bleu','squad']
            fullscores = []
            for j in metrics:
                score = self.test(self.args,i,metric=j,output_dir=output_dir)
                fullscores.append(np.float32(score))
            print(fullscores)
            res.append({
                "lang_pair": i,
                "bleu": fullscores[0],
                "squad": fullscores[1]
            })
        results_df = pd.DataFrame(res)
        results_df.to_csv(os.path.join(output_dir,"results.csv"),index=False)
        args_dict = vars(self.args)
        with open(output_dir+"/args.yaml", 'w') as file:
            yaml.dump(args_dict, file, default_flow_style=False)
            
            
    
        


        
    
    
    
    
    
    
    
        