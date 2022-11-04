import os
import openai
import json
import random

dataset = json.load(open('/data/lyt/exp/single/new/eval.json'))

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

questions = []

for data in dataset[:50]:
    # prompt = '\n'.join( ['Q:' + data['input'] + '\nCOT:' + '\n'.join(data['cot']) + '\nA:' + data['output'] for data in dataset[:-1]])
    prompt = '\n'.join( ['Q:' + data['input'] +  '\nA:' + data['output'] for data in random.choices(dataset, k=5)] )
    questions.append( prompt + '\n'  +'Q:'+ data['input'] + '\nA:') 



response = openai.Completion.create(model="text-davinci-002", prompt=questions, temperature=0, max_tokens=512)

with open('response.json', 'w') as f:
    json.dump(response, f,indent=4,ensure_ascii=False)