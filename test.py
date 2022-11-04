import os
import openai
import json

dataset = json.load(open('question_text.json'))

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = '\n'.join( ['Q:' + data['input'] + '\nCOT:' + '\n'.join(data['cot']) + '\nA:' + data['output'] for data in dataset[:-1]])

prompt = '\n'.join( ['Q:' + data['input'] +  '\nA:' + data['output'] for data in dataset[:-1]])

question = prompt + '\n'  +'Q:'+ dataset[-1]['input'] + '\nA:' 

response = openai.Completion.create(model="text-davinci-002", prompt=question, temperature=0, max_tokens=512)

print(response['choices'][0]['text'])