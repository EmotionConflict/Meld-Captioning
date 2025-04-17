import os
import json
from glob import glob

INPUT_FOLDER = "meld-captioning-pipeline/outputs"
MERGED_JSON = os.path.join(INPUT_FOLDER, "all_data.json")
MERGED_JSONL = os.path.join(INPUT_FOLDER, "all_data.jsonl")

def merge_json_files(input_folder):
    merged_data = []
    for file_path in glob(os.path.join(input_folder, "*.json")):
        if "all_data" in file_path:
            continue
        with open(file_path, "r") as f:
            merged_data.append(json.load(f))
    with open(MERGED_JSON, "w") as f:
        json.dump(merged_data, f, indent=2)
    with open(MERGED_JSONL, "w") as f:
        for entry in merged_data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    merge_json_files(INPUT_FOLDER)