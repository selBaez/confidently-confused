import argparse
import time

import pandas as pd
from tqdm import tqdm

from api import get_response
from constants import CERTAINTY_PREPEND
from utils import num_tokens_from_messages, resolve_language


def read_data(language: int = 0):
    sheet_id = "1H2YImgtNf2gFzheGbwT6xcgsNPLq_YZWqyy9Tia-fpA"
    columns = ["QUESTION", "DOMAIN", "CONFIDENCE", "CONTEXT", "AGREEMENT"]

    df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
    df = df[columns]
    df['PROMPT_TOKENS'] = None
    df['ANSWER'] = None

    print(df.head())

    return df


def write_data(df, language: int = 0):
    language = resolve_language(language)
    df.to_csv(f"./output/answers_{language.lower()}.csv", index=False)


def main(args):
    # Get data
    df = read_data(language=args.language)

    # Prepare storage of labels
    total_tokens = 0
    for i, row in tqdm(df.iterrows()):
        if args.dry_run:
            response = {"text": "DUMMY RUN"}

        else:
            # Pass through model
            try:
                # Ugly hack to adhere to rate limit (1 / s)
                time.sleep(0.4)

                # Call models
                print(row.QUESTION)
                response = get_response(CERTAINTY_PREPEND + row['QUESTION'], api=args.api, model=args.model)
                print(response)
                print("+===========+")

            except Exception:
                response = {"text": "FAILED RUN"}

        # Count tokens to calculate price
        prompt_tokens = num_tokens_from_messages(row.QUESTION,
                                                 model="gpt-3.5-turbo" if args.api == 2 else "text-davinci-002")
        total_tokens += prompt_tokens
        print(f"\t{total_tokens} tokens, {total_tokens * (0.002 / 1000)} dollars")

        # Parse responses
        df.at[i, 'PROMPT_TOKENS'] = prompt_tokens
        df.at[i, 'ANSWER'] = response['text']

    # Write final results to file
    write_data(df, args.language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default=0, type=int, choices=[0, 1, 2, 3, 4],
                        help="Which language slice to use: "
                             "0 English; "
                             "1 German; "
                             "2 Spanish; "
                             "3 Polish; "
                             "4 Dutch; ")
    parser.add_argument('--api', default=2, type=int, choices=[0, 1, 2],
                        help="Which language slice to use: "
                             "0 huggingface; "
                             "1 gpt3; "
                             "2 chatgpt; ")
    parser.add_argument('--model', default='OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5', type=str,
                        help="Which huggingface model to use")
    parser.add_argument('--dry_run', default=True, action='store_true',
                        help="Create the prompts but do not send them to the API")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
