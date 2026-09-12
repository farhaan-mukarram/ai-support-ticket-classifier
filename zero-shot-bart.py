from transformers import pipeline
import numpy as np
from tqdm import tqdm
import pandas as pd

SAMPLE_SIZE = 100

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


def main():
    pipe = pipeline(model="facebook/bart-large-mnli", device=0)

    df = pd.read_csv(
        "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    dev_df = df.sample(n=SAMPLE_SIZE, random_state=42)

    correct = 0

    for row in tqdm(dev_df.iterrows(), desc="Evaluating", total=SAMPLE_SIZE):
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

        if gt_intent == pred:
            correct += 1

    print(f"\tCORRECT = {correct}\n")


if __name__ == "__main__":
    main()
