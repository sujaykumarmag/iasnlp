


# import argparse
# import torch.nn as nn


# import torch
# from src.mbart.configuration_mbart import MBartConfig
# from src.mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
# from src.mbart.tokenization_mbart import MBartTokenizer

# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# from transformers import AlbertTokenizer, AutoTokenizer


# from src.xlnet.modeling_xlnet import XLNetLMHeadModel
# from src.xlnet.configuration_xlnet import XLNetConfig

# from torch import seed
# seed()

# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)


# enc_dec_config = MBartConfig(vocab_size = 64014, bos_token_id= 64000, 
#                      activation_dropout=0.1, attention_dropout=0.1,encoder_layers=2,
#                      decoder_layers=3,pad_token_id=0,eos_token_id=64001)
# enc_dec_model = MBartForConditionalGeneration(config=enc_dec_config)

# print(len(tokenizer))
# print(tokenizer.vocab_size)

# dec_only_config = XLNetConfig(vocab_size = 64014, bos_token_id= 64000, n_layer=6,pad_token_id=0,eos_token_id=64001)
# dec_only_model = XLNetLMHeadModel(config=dec_only_config)

# x = [["Translate the following English sentence to Hindi.\n English sentence: I am a boy\n Hindi sentence:s </s> <2en>"]]
# y = [["<2hi> मैं  एक लड़का हूँ </s>"]]




# inp = tokenizer(x[0], add_special_tokens=False, return_tensors="pt", padding=True).input_ids
# out = tokenizer(y[0], add_special_tokens=False, return_tensors="pt", padding=True).input_ids
# print(inp)
# target_mapping = torch.zeros(
#     (1, 1, inp.shape[1]), dtype=torch.float
# ) 
# output_enc_dec = enc_dec_model(input_ids=inp, labels=out)

# perm_mask = torch.zeros((1, inp.shape[1], inp.shape[1]), dtype=torch.float)
# output_dec = dec_only_model(input_ids=inp, perm_mask=perm_mask, target_mapping=target_mapping, labels=out)


# bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
# eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
# pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

# enc_dec_out = tokenizer.decode(
#     enc_dec_model.generate(inp, use_cache=True, 
#                            num_beams=4, max_length=100, min_length=1, early_stopping=True, pad_token_id=pad_id, 
#                            bos_token_id=bos_id, eos_token_id=eos_id,
#                            decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))[0]
# , skip_special_tokens=True, clean_up_tokenization_spaces=False)

# enc_dec_out = tokenizer.decode(enc_dec_model.generate(inp, max_length=50, num_return_sequences=1)[0], skip_special_tokens=True)

# dec_only_out = tokenizer.decode(dec_only_model.generate(inp, use_cache=True, 
#                            num_beams=4, max_length=100, min_length=1, early_stopping=True, pad_token_id=pad_id, 
#                            bos_token_id=bos_id, eos_token_id=eos_id,
#                            decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))[0], skip_special_tokens=True)

# print("Encoder Model",enc_dec_out)
# print("Decoder only model \n\n",dec_only_out)



# print(enc_dec_model.num_parameters(only_trainable=True))
# print(dec_only_model.num_parameters(only_trainable=True))



import argparse
import torch.nn as nn
import torch
from src.mbart.configuration_mbart import MBartConfig
from src.mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
from src.mbart.tokenization_mbart import MBartTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AlbertTokenizer, AutoTokenizer
from src.xlnet.modeling_xlnet import XLNetLMHeadModel
from src.xlnet.configuration_xlnet import XLNetConfig
from torch import seed

seed()

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

enc_dec_config = MBartConfig(vocab_size=64014, bos_token_id=64000, 
                             activation_dropout=0.1, attention_dropout=0.1, encoder_layers=2,
                             decoder_layers=3, pad_token_id=0, eos_token_id=64001)
enc_dec_model = MBartForConditionalGeneration(config=enc_dec_config)

print(len(tokenizer))
print(tokenizer.vocab_size)

dec_only_config = XLNetConfig(vocab_size=64014, bos_token_id=64000, n_layer=6, pad_token_id=0, eos_token_id=64001)
dec_only_model = XLNetLMHeadModel(config=dec_only_config)

x = [["Translate the following English sentence to Hindi.\n English sentence: I am a boy\n Hindi sentence:s </s> <2en>"]]
y = [["<2hi> मैं  एक लड़का हूँ </s>"]]

inp = tokenizer(x[0], add_special_tokens=False, return_tensors="pt", padding=True).input_ids
out = tokenizer(y[0], add_special_tokens=False, return_tensors="pt", padding=True).input_ids

print("Input shape:", inp.shape)
print("Output shape:", out.shape)

# target_mapping = torch.zeros(
#     (1, 1, inp.shape[1]), dtype=torch.float
# # ) 

# output_enc_dec = enc_dec_model(input_ids=inp, labels=out)

perm_mask = torch.zeros((1, inp.shape[1], inp.shape[1]), dtype=torch.float)

# output_dec = dec_only_model(input_ids=inp, labels=out)

bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

enc_dec_out = tokenizer.decode(
    enc_dec_model.generate(inp, use_cache=True, 
                           num_beams=4, max_length=100, min_length=1, early_stopping=True, pad_token_id=pad_id, 
                           bos_token_id=bos_id, eos_token_id=eos_id,
                           decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))[0]
, skip_special_tokens=True, clean_up_tokenization_spaces=False)

enc_dec_out = tokenizer.decode(enc_dec_model.generate(inp, max_length=50, num_return_sequences=1)[0], skip_special_tokens=True)

dec_only_out = tokenizer.decode(dec_only_model.generate(inp, use_cache=True, 
                           num_beams=4, max_length=100, min_length=1, early_stopping=True, pad_token_id=pad_id, 
                           bos_token_id=bos_id, eos_token_id=eos_id,
                           decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))[0], skip_special_tokens=True)

print("Encoder Model", enc_dec_out)
print("Decoder only model \n\n", dec_only_out)

print(enc_dec_model.num_parameters(only_trainable=True))
print(dec_only_model.num_parameters(only_trainable=True))

