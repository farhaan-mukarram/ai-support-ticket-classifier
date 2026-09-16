import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from benchmarks import benchmark_bart, benchmark_qwen

SAMPLE_SIZE = 1000
SEED = 42

OUTPUT_PATH = "res"


def main(df: DataFrame):
    output_dir = Path(OUTPUT_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)

    report, elapsed_time = benchmark_bart(df)

    with open(f"{output_dir}/bart.json", "w") as fp:
        report["time"] = elapsed_time
        json.dump(report, fp)

    report, elapsed_time = benchmark_qwen(df)

    with open(f"{output_dir}/qwen.json", "w") as fp:
        report["time"] = elapsed_time
        json.dump(report, fp)


if __name__ == "__main__":
    df = pd.read_csv(
        "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )

    df = df.sample(SAMPLE_SIZE, random_state=SEED)

    main(df)
