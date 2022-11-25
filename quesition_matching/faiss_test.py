import numpy as np
import json
import os
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
dim = 768

input_path = '/data/lyt/exp/single/company'
output_file = 'match_idx_company.json'
with open('question_embs_company.npy','rb') as f:
    question_embs = np.load(f)

train_dataset = json.load(open(os.path.join(input_path,'train.json')))
test_dataset = json.load(open(os.path.join(input_path,'test.json')))
# test_dataset = json.load(open('probs.json'))


import faiss

res = faiss.StandardGpuResources()  # use a single GPU
index_flat = faiss.IndexFlatL2(dim)  # build a flat (CPU) index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 2, index_flat)
gpu_index_flat.add(question_embs)         # add vectors to the index

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

k = 10
match_idx = []
for data in tqdm(test_dataset[:]):
    encoded_input = tokenizer(
        text=data['context'],
        text_pair=data['question'],
        return_tensors='pt',
        max_length=512,
        truncation='only_first',
        padding='max_length'
    )
    output = model(**encoded_input)
    D, I = gpu_index_flat.search(output[1].detach().numpy(), k)
    match_idx.append(I[0].tolist())
with open(output_file,'w') as f:
    json.dump(match_idx,f,indent=4)