import os

from dotenv import load_dotenv
from fastapi import FastAPI
from ollama import ChatResponse, chat
from pydantic import BaseModel, ValidationError
from slack_sdk.webhook import WebhookClient

from backend.database_utils import insert_into_table
from constants import INTENTS_LIST, Ticket

load_dotenv()

url = os.environ.get("SLACK_WEBHOOK_URL")
webhook = WebhookClient(url)


class Item(BaseModel):
    description: str


def send_slack_alert(ticket_id: int | str, description: str, intent: str):
    blocks = [
        {
            "type": "card",
            "title": {
                "type": "mrkdwn",
                "text": "🚨 Ticket flagged for human review 🚨.",
                "verbatim": False,
            },
            "body": {
                "type": "mrkdwn",
                "text": f"Ticket ID: '{ticket_id}', description: '{description}', intent: '{intent}'",
                "verbatim": False,
            },
        },
    ]

    webhook.send(
        text=f"🚨 Ticket flagged for human review 🚨. Ticket ID: '{ticket_id}', description: '{description}', intent: '{intent}'",
        blocks=blocks,
    )


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Bye World"}


@app.post("/api/classify/")
async def classify(item: Item) -> str | None:
    description = item.description

    response: ChatResponse = chat(
        model="qwen3.5:2b",
        messages=[
            {
                "role": "user",
                "content": f"Classify the ticket into intent based on the description. Here is the list of allowed intents: {INTENTS_LIST}. Here is the description to classify: {description}. Return the description and most likely intent.",
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

        # write results to db
        ticket_id = insert_into_table(description=description, intent=res.intent)

        # flag for human review and generate slack alert
        match res.intent:
            case "contact_human_agent" | "manual_review":
                send_slack_alert(ticket_id, description, intent=res.intent)

        return res.intent

    return None
