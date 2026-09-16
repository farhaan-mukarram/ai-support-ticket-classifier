import time

import numpy as np
from ollama import ChatResponse, chat
from pandas import DataFrame
from pydantic import ValidationError
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import pipeline

from constants import INTENTS_LIST, Ticket


def benchmark_bart(df: DataFrame):
    pipe = pipeline(model="facebook/bart-large-mnli", device=0)

    y_pred = []
    y_true = []
    correct = 0

    time_start = time.perf_counter()

    for row in tqdm(df.iterrows(), desc="Evaluating Bart...", total=len(df)):
        item = row[-1]

        desc: str = item["instruction"]
        gt_intent: str = item["intent"]

        res = pipe(
            desc,
            candidate_labels=INTENTS_LIST,
        )

        scores = res["scores"]
        labels = res["labels"]

        pred = labels[np.argmax(scores)]

        y_true.append(gt_intent)
        y_pred.append(pred)

        if gt_intent == pred:
            correct += 1

    time_end = time.perf_counter()
    elapsed = time_end - time_start

    report = classification_report(
        y_true, y_pred, labels=INTENTS_LIST, output_dict=True
    )

    return report, elapsed


def benchmark_qwen(df: DataFrame):
    y_pred = []
    y_true = []
    correct = 0

    invalid = 0

    time_start = time.perf_counter()

    for row in tqdm(df.iterrows(), desc="Evaluating Qwen...", total=len(df)):
        item = row[-1]

        desc: str = item["instruction"]
        gt_intent: str = item["intent"]

        response: ChatResponse = chat(
            model="qwen3.5:2b",
            messages=[
                {
                    "role": "user",
                    "content": f"Classify the ticket into intent based on the description. Here is the list of allowed intents: {INTENTS_LIST!s}. Here is the description to classify: {desc}. Return the description and most likely intent.",
                }
            ],
            format=Ticket.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
            options={"temperature": 0},  # Make responses more deterministic
            think=False,
        )

        if response.message.content is not None:
            try:
                # Use Pydantic to validate the response
                res = Ticket.model_validate_json(response.message.content)
                y_true.append(gt_intent)
                y_pred.append(res.intent)

            except ValidationError:
                res = {"description": desc, "intent": "manual_review"}
                invalid += 1

            finally:
                if res is not None and gt_intent == res.intent:
                    correct += 1

    time_end = time.perf_counter()
    elapsed = time_end - time_start

    report = classification_report(
        y_true, y_pred, target_names=INTENTS_LIST, output_dict=True
    )

    return report, elapsed
