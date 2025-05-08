import pandas as pd
import json

# Load paths to your input files
csv_path = "test_labels_subset.csv"  # replace with your actual path
json_path = "revised_final_annotations.json"  # replace with your actual path
output_json_path = "updated_annotations.json"  # desired output file name

# Step 1: Load the CSV file
csv_df = pd.read_csv(csv_path)

# Step 2: Create a new column 'video_id' to match the JSON format
csv_df["video_id"] = csv_df.apply(
    lambda row: f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}.mp4", axis=1
)

# Step 3: Create a dictionary for fast lookup: video_id -> {Utterance, Emotion}
utterance_lookup = {
    row["video_id"]: {"Utterance": row["Utterance"], "Emotion": row["Emotion"]}
    for _, row in csv_df.iterrows()
}

# Step 4: Load the JSON file
with open(json_path, 'r') as f:
    json_data = json.load(f)

# Step 5: Append 'Utterance' and 'Emotion' to each JSON item if video_id matches
for item in json_data:
    video_id = item.get("video_id")
    if video_id in utterance_lookup:
        item.update(utterance_lookup[video_id])

# Step 6: Save the updated JSON to a new file
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON saved to: {output_json_path}")
