from typing import Literal

from fastapi import FastAPI
from ollama import ChatResponse, chat
from pydantic import BaseModel, ValidationError

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


class Item(BaseModel):
    description: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Bye World"}


@app.post("/classify/")
async def classify(item: Item) -> str | None:
    description = item.description

    response: ChatResponse = chat(
        model="qwen3.5:0.8b",
        messages=[
            {
                "role": "user",
                "content": f"Classify the ticket into intent based on the description. Here is the list of allowed intents: {INTENTS_LIST!s}. Here is the description to classify: {description}. Return the description and most likely intent.",
            }
        ],
        format=Ticket.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
        options={"temperature": 0},  # Make responses more deterministic
        think=False,
    )

    if response.message.content is not None:
        res = None

        try:
            # Use Pydantic to validate the response
            res = Ticket.model_validate_json(response.message.content)

        except ValidationError:
            res = {"description": description, "intent": "manual_review"}

        return res.intent

    return None
