import pandas as pd
from typing import Literal
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from ollama import chat, ChatResponse

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


def main():

    df = pd.read_csv(
        "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    dev_df = df.sample(n=SAMPLE_SIZE, random_state=42)

    for model in ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"]:
        print(f"MODEL = {model.upper()}")

        correct = 0

        # generation errors
        invalid = 0

        for row in tqdm(dev_df.iterrows(), desc="Evaluating", total=SAMPLE_SIZE):
            item = row[-1]

            desc: str = item["instruction"]
            gt_intent: str = item["intent"]

            response: ChatResponse = chat(
                model=model,
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
                except ValidationError as e:
                    res = {"description": desc, "intent": "manual_review"}
                    invalid += 1

                finally:
                    if res is not None and gt_intent == res.intent:
                        correct += 1

        print(f"\tCORRECT = {correct}\n, INVALID = {invalid}")


if __name__ == "__main__":
    main()
