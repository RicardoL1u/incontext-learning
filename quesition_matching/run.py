import openai
import numpy as np
import json
import os
import requests


input_path = '/data/lyt/exp/single/new'
train_dataset = json.load(open(os.path.join(input_path,'train.json')))
test_dataset = json.load(open(os.path.join(input_path,'test.json')))

openai.api_key = "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"
GLM_URL = '103.238.162.37:9622'


with open('match_idx.json','rb') as f:
    match_idx_list = json.load(f)

k=3
for tgt,match_idx in zip(range(len(test_dataset)),match_idx_list):
    prompt = '\n'.join([train_dataset[idx]['input'] + '\n' + train_dataset[idx]['output'] for idx in match_idx[:k]])
    prompt += 'Let\'s think step by step\n'
    prompt += test_dataset[tgt]['input']
    print(prompt)
    # print()
    # response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=0, max_tokens=512)
    # print(response['choices'][0]['text'])
    
    response = requests.post(GLM_URL,json={'seed':42,'contexts':[prompt]})
    print(response)
    break


# import os
# import openai
# import json

# dataset = json.load(open('question_text.json'))

# # Load your API key from an environment variable or secret management service

# prompt = '\n'.join( ['Q:' + data['input'] + '\nCOT:' + '\n'.join(data['cot']) + '\nA:' + data['output'] for data in dataset[:-1]])

# prompt = '\n'.join( ['Q:' + data['input'] +  '\nA:' + data['output'] for data in dataset[:-1]])

# question = prompt + '\n'  +'Q:'+ dataset[-1]['input'] + '\nA:' 


# print(response['choices'][0]['text'])