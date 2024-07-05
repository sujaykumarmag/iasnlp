import argparse
import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer
from src.xlnet.modeling_xlnet import XLNetLMHeadModel
from src.xlnet.configuration_xlnet import XLNetConfig


from src.mbart.configuration_mbart import MBartConfig
from src.mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
from src.mbart.tokenization_mbart import MBartTokenizer




# Seed for reproducibility
torch.manual_seed(42)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)



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

# Input and Output sequences
x = "I am a boy </s> <2en>"
y = "<2hi> मैं एक लड़का हूँ </s>"

# Tokenize inputs and outputs
inp = tokenizer(x, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
out = tokenizer(y, add_special_tokens=False, return_tensors="pt", padding=True).input_ids

# Model Outputs
output_enc_dec = enc_dec_model(input_ids=inp, decoder_input_ids=out[:, :-1], labels=out[:, 1:])
output_dec = dec_only_model(input_ids=inp, labels=out)



















# Token IDs
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

# Generate and decode outputs
enc_dec_out = tokenizer.decode(
    enc_dec_model.generate(
        inp, use_cache=True, num_beams=4, max_length=20, min_length=1, early_stopping=True,
        pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id,
        decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>")
    )[0],
    skip_special_tokens=True, clean_up_tokenization_spaces=False
)

dec_only_out = tokenizer.decode(
    dec_only_model.generate(inp, max_length=50, num_return_sequences=1)[0],
    skip_special_tokens=True
)

# Print generated outputs
print("Encoder-Decoder Model Output:", enc_dec_out)
print("Decoder-Only Model Output:\n\n", dec_only_out)

# Print number of trainable parameters
print("Encoder-Decoder Model Trainable Parameters:", enc_dec_model.num_parameters(only_trainable=True))
print("Decoder-Only Model Trainable Parameters:", dec_only_model.num_parameters(only_trainable=True))
