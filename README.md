# Multilingual Neural Machine Translation (NMT) Dataset for In-context Learning, Finetuning, and Baseline Model Development



## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Installation](#installation)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Pending Tasks](#pending-tasks)
- [Cite This Work](#cite-this-work)
- [References](#references)

## Problem Statement

Multilingual Neural Machine Translation (NMT) enables training a single model capable of translating between multiple source and target languages. Traditional approaches use encoder-decoder architectures, while recent advancements explore the use of Large Language Models (LLMs) for Multilingual Machine Translation (MMT). This project investigates:

1. **Performance Comparison:** Evaluate the performance of encoder-decoder based MT versus smaller LLMs trained on the same data with similar parameters.
   
2. **Context Role Quantification:** Analyze the impact of context (number of tokens) on translation quality for both architectures.


## Dataset

The dataset provided includes:
- One-to-One translations
- One-to-Many translations
- Many-to-One translations


- **MT Dataset**: Contains data necessary for training and evaluation across various translation scenarios.
- **Google Drive Link**: [MT Dataset and Results](https://drive.google.com/drive/folders/1rvzWJAMYXlZLI2l_FyIZ77jHNwaWTMFr?usp=sharing)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sujaykumarmag/iasnlp.git
   cd DSP
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Configuration parameters will be saved in `runs/args.yaml` for each experiment

### Example `args.yaml`
```yaml
batchsize: 10
cross_lang: false
direction_order: null
disorder: false
ex_lang: null
experiment: icl
ice_num: 8
lang_order: null
lang_pair: eng-hin
lora_alpha: 16
lora_dropout: 0.05
lr: 0.001
many2one: false
model_name: facebook/xglm-564M
model_type: enc_dec
multi: eng hin mar
numepochs: 5
one2many: false
oracle: false
output_dir: runs/
prompt_template: </E></X>=</Y>
repeat: false
retriever: random
reverse_direction: false
run_all_icl: true
seed: 43
tokenizer_name: google/mt5-base


```
   
## File Structure
```bash

root_directory/
├── src/
│   ├── baselines.py
│   │
│   ├── dataset.py
│   │
│   ├── decoder_only.py
│   │
│   ├── icl.py
│   │
│   ├── utils.py
│   │
│   ├── mbart/ (from hugging face)
│   │   ├── configuration_mbart.py
│   │   ├── modeling_mbart.py 
│   │   └── tokenization_mbart.py
│   │
│   │
│   ├── xlnet/ (from hugging face)
│   │   ├── configuration_xlnet.py
│   │   ├── modeling_xlnet.py 
│   │   └── tokenization_xlnet.py
│   │
│   │
│   ├── training/
│       ├── normal_train.py
│       └── training.py
│  
├── finetuning\ (all notebooks ran on kaggle)
│  
├── runs/
│   ├── exp1
│   ├── exp2
│   ├── exp3
│   ├── exp4
│   ├── exp5 
│   ├── exp6
│   ├── exp7
│   ├── exp8
│   └── exp9
│   
│   
├── train.py  # (Entry Point for the Program)
│   
├── trainall.bash
└── notebooks/
    └── train.ipynb
```

## Demo Video

https://github.com/sujaykumarmag/iasnlp/assets/75253527/b9883a1d-7132-498a-8d2e-c92217f150d9



## Pending Tasks

- [x] Include Finetuning Code
- [x] Enhance documentation with more detailed explanations (report in `IASNLP_Project_report.pdf`)
- [ ] Add support for GPU training (MPS Not available in TorchDrug)
- [ ] Research on SSA Attention Method



## References

- [Massively Multilingual Neural Machine Translation](https://aclanthology.org/N19-1388.pdf)
- [Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis](https://arxiv.org/pdf/2304.04675)


## Cite this Work


```
@misc{iasnlp_project,
  author = {Abhinav P.M ., SujayKumar Reddy M., Oswald.C (Machine Translators)},
  title = {In-context Learning (ICL), Finetuning and Baseline Model Development for Natural Machine Translation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sujaykumarmag/iasnlp}},
}
```

