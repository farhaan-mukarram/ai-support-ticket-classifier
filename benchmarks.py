from enum import Enum
import time

from transformers import pipeline
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from typing import Literal
from pydantic import BaseModel, ValidationError
from ollama import chat, ChatResponse

from sklearn.metrics import classification_report

INTENTS_LIST = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
]


# Define the schema for the response
class Ticket(BaseModel):
    description: str
    # TODO: Find a cleaner way to do this
    intent: Literal[
        "cancel_order",
        "change_order",
        "change_shipping_address",
        "check_cancellation_fee",
        "check_invoice",
        "check_payment_methods",
        "check_refund_policy",
        "complaint",
        "contact_customer_service",
        "contact_human_agent",
        "create_account",
        "delete_account",
        "delivery_options",
        "delivery_period",
        "edit_account",
        "get_invoice",
        "get_refund",
        "newsletter_subscription",
        "payment_issue",
        "place_order",
        "recover_password",
        "registration_problems",
        "review",
        "set_up_shipping_address",
        "switch_account",
        "track_order",
        "track_refund",
    ]


class Model(Enum):
    BART = "facebook/bart-large-mnli"
    QWEN = "qwen3.5:0.8b"


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
            model="qwen3.5:0.8b",
            messages=[
                {
                    "role": "user",
                    "content": f"Classify the ticket into intent based on the description. Here is the list of allowed intents: {str(INTENTS_LIST)}. Here is the description to classify: {desc}. Return the description and most likely intent.",
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

            except ValidationError as e:
                res = {"description": desc, "intent": "manual_review"}
                invalid += 1

            finally:
                if res is not None and gt_intent == res.intent:
                    correct += 1

    time_end = time.perf_counter()
    elapsed = time_end - time_start

    report = classification_report(
        y_true, y_pred, labels=INTENTS_LIST, output_dict=True
    )

    return report, elapsed
