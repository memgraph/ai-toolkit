import argparse
import csv

from mcp_prompt_lib.prompt_lib import ask_with_tools


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt with tools from a CSV file.")
    parser.add_argument("csv_file_path", help="Path to the CSV file containing prompts.")
    args = parser.parse_args()

    prompts = {}
    with open(args.csv_file_path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts[row["id"]] = row["prompt"]
    for id, prompt in prompts.items():
        print("--------------------------------")
        answer = ask_with_tools(prompt, model="ollama/llama3.2")
        print(f"'{id}': {prompt}")
        print(f"Answer: {answer}")
        print("--------------------------------")