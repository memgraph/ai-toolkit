import csv
import argparse

parser = argparse.ArgumentParser(
    description="Convert a text file to a CSV file with one line per row."
)
parser.add_argument("--txt", required=True, help="Path to the input text file.")
parser.add_argument("--csv", required=True, help="Path to the output CSV file.")
args = parser.parse_args()

txt_file_path = args.txt
csv_file_path = args.csv
with (
    open(txt_file_path, "r", encoding="utf-8") as txt_file,
    open(csv_file_path, "w", encoding="utf-8") as csv_file,
):
    fieldnames = ["id", "prompt"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i, line in enumerate(txt_file):
        writer.writerow({"id": i, "prompt": line.rstrip("\n")})
