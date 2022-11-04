import json
import random
import requests
dataset = [data[0] for data in json.load(open("result.json"))]


prompt = ''
samples = random.choices(dataset, k=8)
for data in samples[:-1]:
    prompt += '上下文：' + data['context'] + '\n'
    prompt += '问题：' + data['question'] + '?\n'
    prompt += '让我们一步步思考！先从上下文中找到证据再找答案！\n'
    prompt += '推理证据：' + ''.join(data['rationale_text']) + '\n'
    prompt += '答案：' + data['ans'] + '\n\n'

data = samples[-1]
prompt += '上下文：' + data['context'] + '\n'
prompt += '问题：' + data['question'] + '?\n'
prompt += '让我们一步步思考！先从上下文中找到证据再找答案！\n'
prompt += '推理证据：'
print(prompt)

url = '103.238.162.37:9622'
myobj = {
    'seed': 43,
    'contexts':[prompt],
}

x = requests.post(url, json = myobj)
print(x)