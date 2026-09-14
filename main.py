import pandas as pd
from pandas import DataFrame
from benchmarks import benchmark_bart, benchmark_qwen

from pathlib import Path
import json

SAMPLE_SIZE = 1000
SEED = 42

OUTPUT_PATH = "res"


def main(df: DataFrame):
    output_dir = Path(OUTPUT_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)

    report, elapsed_time = benchmark_bart(df)

    report["time"] = elapsed_time

    with open(f"{output_dir}/bart.json", "w") as fp:
        json.dump(report, fp)

    report, elapsed_time = benchmark_qwen(df)

    report["time"] = elapsed_time

    with open(f"{output_dir}/qwen.json", "w") as fp:
        json.dump(report, fp)


if __name__ == "__main__":
    df = pd.read_csv(
        "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )

    df = df.sample(SAMPLE_SIZE, random_state=SEED)

    main(df)
