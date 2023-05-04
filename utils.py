from typing import Union

import tiktoken


def resolve_language(language: int = 0):
    if language == 0:
        return "English"
    elif language == 1:
        return "German"
    elif language == 2:
        return "Spanish"
    elif language == 3:
        return "Polish"
    elif language == 4:
        return "Dutch"


def num_tokens_from_messages(messages: Union[list, str], model: str = "gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        messages = [{"role": "user", "content": messages}]
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
