import openai
import numpy as np
import json
import os
import requests
from tqdm import tqdm

k=1
output_file = f'glm_result_company_k={k}.json'
match_file = 'match_idx_company.json'

input_path = '/data/lyt/exp/single/company'
train_dataset = json.load(open(os.path.join(input_path,'train.json')))
test_dataset = json.load(open(os.path.join(input_path,'test.json')))
# test_dataset = json.load(open('probs.json'))



openai.api_key = "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"
GLM_URL = 'http://103.238.162.37:9622'

def get_prompt(match_idx:list,k:int):
    prompt = '\n'.join(['context:'+train_dataset[idx]['context'] + 
        '\n' + 'question:'+train_dataset[idx]['question']  + 
        '\n' + 'answers:' +' <ans> '.join(train_dataset[idx]['answers'])+' <stop>' for idx in match_idx[:k] ] )
    prompt += '\nLet\'s think step by step\n'
    return prompt
with open(match_file,'rb') as f:
    match_idx_list = json.load(f)


result = []
for tgt,match_idx in tqdm(zip(range(len(test_dataset)),match_idx_list),total=len(test_dataset)):
    prompt = get_prompt(match_idx,k)
    prompt +=  'context:' + test_dataset[tgt]['context'] + '\n'  + 'question:' + test_dataset[tgt]['question'] + '\n' + 'answers:'
    # print(prompt)
    # print()
    # response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=0, max_tokens=512)
    # print(response['choices'][0]['text'])
    
    response = requests.post(GLM_URL,json={'seed':42,'contexts':[prompt]})
    test_dataset[tgt]['pred'] = json.loads(response.text)['output'][0]
    test_dataset[tgt]['preds'] = test_dataset[tgt]['pred'].split('<stop>')[0].strip().split(' <ans> ') if '<stop>' in test_dataset[tgt]['pred'] else []
    # test_dataset[tgt]['prompt'] = get_prompt(match_idx,k)
    result.append(test_dataset[tgt])
    if tgt == 1000:
        break


with open(output_file,'w') as f:
    json.dump(result,f,indent=4,ensure_ascii=False)


# import os
# import openai
# import json

# dataset = json.load(open('question_text.json'))

# # Load your API key from an environment variable or secret management service

# prompt = '\n'.join( ['Q:' + data['input'] + '\nCOT:' + '\n'.join(data['cot']) + '\nA:' + data['output'] for data in dataset[:-1]])

# prompt = '\n'.join( ['Q:' + data['input'] +  '\nA:' + data['output'] for data in dataset[:-1]])

# question = prompt + '\n'  +'Q:'+ dataset[-1]['input'] + '\nA:' 


# print(response['choices'][0]['text'])