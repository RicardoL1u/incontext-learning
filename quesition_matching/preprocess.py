import json
import argparse
import os
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
# bsz = 16
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.to('cuda:1')
# parser = argparse.ArgumentParser()
# parser.add_argument('--input','-i',required=True,help='plz provide input path')
# args = parser.parse_args()

# train_dataset = json.load(open(os.path.join(args.input,'train.json')))
question_embs = []
train_dataset = json.load(open(os.path.join('/data/lyt/exp/single/company','train.json')))
for data in tqdm(train_dataset[:]):
    # print(data['question'])
    encoded_input = tokenizer(
        text = data['context'],
        text_pair = data['question'],
        max_length=512,
        padding='max_length',
        truncation='only_first',
        return_tensors='pt'
    )
    # print(encoded_input)
    # print(tokenizer.batch_decode(encoded_input['input_ids']))
    for k,_ in encoded_input.items():
        encoded_input[k]=encoded_input[k].to(torch.device('cuda:1'))
    # print(encoded_input)
    output = model(**encoded_input)
    question_embs.append(output[1].cpu().detach().numpy())
    # break
question_embs = np.concatenate(question_embs,axis=0).reshape((-1,768))
# print(question_embs.shape)
with open('question_embs_company.npy','wb') as f:
    np.save(f,question_embs)


