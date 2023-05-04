import json
import requests
import openai

from dotenv import dotenv_values
from utils import gpt3, chatgpt

CONFIG = dotenv_values(".env")
AVAILABLE_APIS = [
     'huggingface',
     'gpt3',
     'chatgpt',
]

def query(payload, headers, url):
        data = json.dumps(payload)
        response = requests.request("POST", url, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

def call_api_hf(prompt, model="gpt2"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {CONFIG['HF_API_TOKEN']}"}
    data = query(prompt, headers, url)
    return data

def get_response(prompt, api="huggingface", model="gpt2"):

    print(f"Getting response from API: {api} with model={model}")
    if api == "huggingface":
        response = call_api_hf(prompt, model=model)
    elif api == 'gpt3':
        openai.api_key = CONFIG['OPENAI_API_TOKEN']
        response = gpt3([prompt], model=model)
    elif api == 'chatgpt':
        openai.api_key = CONFIG['OPENAI_API_TOKEN']
        response = chatgpt([prompt], model=model)
    else:
        raise ValueError(f"API {api} not available. Available APIs: {AVAILABLE_APIS}")

    print(response)
    return response



if __name__ == "__main__":
    question = "Who is the most important Mexican president? (m)"
    get_response(question)
    get_response(question, api="gpt3", model="text-davinci-002")

