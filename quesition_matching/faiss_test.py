import numpy as np
import json
import os
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
dim = 768

input_path = '/data/lyt/exp/single/new'
with open('question_embs.npy','rb') as f:
    question_embs = np.load(f)

train_dataset = json.load(open(os.path.join(input_path,'train.json')))
test_dataset = json.load(open(os.path.join(input_path,'test.json')))


import faiss

res = faiss.StandardGpuResources()  # use a single GPU
index_flat = faiss.IndexFlatL2(dim)  # build a flat (CPU) index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 2, index_flat)
gpu_index_flat.add(question_embs)         # add vectors to the index

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

k = 10
match_idx = []
for data in tqdm(test_dataset[:10]):
    encoded_input = tokenizer(data['input'],
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )
    output = model(**encoded_input)
    D, I = gpu_index_flat.search(output[1].detach().numpy(), k)
    match_idx.append(I[0].tolist())
print(match_idx)
print(len(match_idx))
# match_idx = np.array(match_idx)
with open('match_idx.json','w') as f:
    json.dump(match_idx,f,indent=4)