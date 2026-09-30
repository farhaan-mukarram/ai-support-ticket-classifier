# AI-Powered Intent Classification

An automated triage system using local LLMs to classify customer support tickets into 20+ categories, reducing manual overhead and providing real-time Slack alerts for human review.

## Benchmarks
A comparative analysis was conducted to select the optimal model:

| Metric | LLM-based (Qwen 3.5 2B) | Zero-Shot NLI (BART) |
| :--- | :--- | :--- |
| Accuracy (F1 Score) | 0.72 | 0.33 |
| Latency | 329ms | 221ms |

The Qwen 3.5 2B model was selected due to its superior performance for the task, despite the increase in latency.

## Key Features
- **Schema Enforcement**: Uses Pydantic to ensure strict JSON outputs, to reduce LLM hallucinations.
- **Automated Escalation**: Integrates Slack Webhooks to alert human agents in real-time for specific intents.
- **Persistent Logging**: Automatically stores all classifications in a SQLite database for analytics.

## Tech Stack
- **AI/ML**: Ollama, HuggingFace Transformers, Pandas, Scikit-learn
- **Backend**: Python, FastAPI, Pydantic
- **Infrastructure**: SQLite, Slack SDK
