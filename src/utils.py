

import os
import re


def get_contents(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            sentences.append(sentence)
    return sentences

def get_duals(datatypes,lang,PATH):
    for i in datatypes:
        x = os.listdir(os.path.join(PATH,i))
        x.remove("domain.txt")
        x.remove("train.eng_Latn")
        if lang in x[0]:
            res_lang = get_contents(os.path.join(PATH,i,x[0]))
            res_en = get_contents(os.path.join(PATH,i,"train.eng_Latn"))
            return {lang:res_lang,"en":res_en}
        
        
        

def remove_file_artifacts(arr):
    try:
        arr.remove(".DS_Store")
    except:
        print("No .DS_Store file found !!!")
    return arr



def get_exp(output_dir="runs/"):
    
    try:
        exps = os.listdir(output_dir)
    except:
        os.makedirs(output_dir)
        exps = os.listdir(output_dir)
        
    exps = remove_file_artifacts(exps)
    exp_numbers = sorted([int(re.search(r'\d+', exp).group()) for exp in exps if re.search(r'\d+', exp)])
    
    next_number = exp_numbers[-1] + 1 if exp_numbers else 1
    next_exp = f"exp{next_number}"
    
    new_dir = os.path.join(output_dir, next_exp)
    os.makedirs(new_dir,exist_ok=True)
    return new_dir
    
    
    