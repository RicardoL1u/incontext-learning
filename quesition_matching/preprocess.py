import json
import argparse
import os
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
bsz = 16
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.to('cuda:1')
# parser = argparse.ArgumentParser()
# parser.add_argument('--input','-i',required=True,help='plz provide input path')
# args = parser.parse_args()

# train_dataset = json.load(open(os.path.join(args.input,'train.json')))
question_embs = []
train_dataset = json.load(open(os.path.join('/data/lyt/exp/single/new','train.json')))

for i in tqdm(range(0,len(train_dataset),bsz)):
    encoded_input = tokenizer(
        [data['input'] for data in train_dataset[i:i+bsz]],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    for k,_ in encoded_input.items():
        encoded_input[k]=encoded_input[k].to(torch.device('cuda:1'))
    # print(encoded_input)
    output = model(**encoded_input)
    question_embs.append(output[1].cpu().detach().numpy())
    # break
question_embs = np.concatenate(question_embs,axis=0).reshape((-1,768))

with open('question_embs.npy','wb') as f:
    np.save(f,question_embs)


