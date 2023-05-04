from typing import Union

import openai
import tiktoken

EXAMPLE_SEPARATOR = "\n\n-----------------------\n\n"


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


def num_tokens_from_messages(messages: Union[list, str], model: str = "gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    if model == "text-davinci-002":
        num_tokens = len(encoding.encode(messages))
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")
