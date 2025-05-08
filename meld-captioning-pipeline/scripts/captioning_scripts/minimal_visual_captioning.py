import sys
import os
import io
import json
import base64
from PIL import Image
import cv2
from openai import OpenAI
import traceback

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AU_labeling.au_extraction_new import extract_au_from_video, parse_au_intensity
from AU_labeling.peak_frame_description_new import find_peak_frame, extract_frame_by_index

# Configs
video_file = "dia12_utt12.mp4"
video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MELD_test_subset/test_subset", video_file))
subtitle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MELD_test_subset/test_subtitles", video_file.replace(".mp4", ".txt")))
au_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../AU_labeling/AU_data/{video_file.replace('.mp4', '.csv')}"))
output_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MELD_test_subset/MELD_annotations.json"))
openface_bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../OpenFace/build/bin/FeatureExtraction"))

client = OpenAI(api_key="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") #"API-KEY"

try:
    print(f"[INFO] Processing {video_path}")
    
    os.makedirs(os.path.dirname(au_csv_path), exist_ok=True)
    extract_au_from_video(video_path, os.path.dirname(au_csv_path), openface_bin_path)
    print("[INFO] AU extraction done")

    peak_frame, time = find_peak_frame(au_csv_path)
    au_phrases, raw_aus = parse_au_intensity(au_csv_path, peak_frame)
    print("[INFO] Peak frame:", peak_frame)

    image = extract_frame_by_index(video_path, peak_frame)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the scene."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        max_tokens=500
    )
    visual_caption = response.choices[0].message.content

    with open(subtitle_path, "r") as f:
        subtitle = f.read()

    result = {
        "video_id": video_file,
        "peak_time": time,
        "visual_expression_description": au_phrases,
        "visual_objective_description": visual_caption,
        "raw_AU_values_at_peak": raw_aus,
        "coarse-grained_summary": f"{visual_caption} Saying: '{subtitle}'"
    }

    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(result)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✔] Saved to {output_json_path}")

except Exception as e:
    print("[✖ ERROR]")
    print(traceback.format_exc())
