import json

import openai
import requests
from dotenv import dotenv_values

from constants import CERTAINTY_PREPEND

CONFIG = dotenv_values(".env")
AVAILABLE_APIS = [
    'huggingface',
    'gpt3',
    'chatgpt',
]


def query(payload, headers, url):
    data = json.dumps(payload)
    response = requests.request("POST", url, headers=headers, data=data)
    if response.status_code != 200:
        print(payload)
        print(response)
    response_data = json.loads(response.content.decode("utf-8"))
    return response_data


def call_api_hf(prompt, model="gpt2"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CONFIG['HF_API_TOKEN']}"
    }
    data = query({'inputs': prompt, 'options': {'wait_for_model': True}}, headers, url)
    return data


def gpt3(prompts: list, model: str = "text-davinci-002"):
    """ functions to call GPT3 predictions, works on batches """
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return [
        {
            'prompt': prompt,
            'text': response['text'],
            'tokens': response['logprobs']['tokens'],
            'logprob': response['logprobs']['token_logprobs']
        }
        for response, prompt in zip(response.choices, prompts)
    ]


def chatgpt(utterances: list, model: str = "gpt-3.5-turbo"):
    """ functions to call ChatGPT predictions """
    response = openai.ChatCompletion.create(
        model=model,
        messages=utterances,
        temperature=0,
        top_p=1,
        max_tokens=100,  # Manually change to 60 with prompt style 1
        frequency_penalty=0,
        presence_penalty=0
    )
    return {
        'prompt': utterances,
        'text': response['choices'][0]['message']['content'],
        'finish_reason': response['choices'][0]['finish_reason']
    }


def get_response(prompt, api="huggingface", model="gpt2"):
    prompt = CERTAINTY_PREPEND + prompt

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

    # remove line from response
    response = response[0]['generated_text'].replace(CERTAINTY_PREPEND, "")

    return response


if __name__ == "__main__":
    with open("questions.txt", "r") as f:
        for line in f:
            print(line)
            response = get_response(line, model='OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5')
            print(response)
            print("+===========+")
