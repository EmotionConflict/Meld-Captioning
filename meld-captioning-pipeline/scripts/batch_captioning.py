# Batch captioning script for MELD dataset
# Load each video/audio/subtitle
# Extract middle frame
# Run BLIP-2 for frame caption
# Run Whisper for audio transcription
# Merge everything into one paragraph
# Save into meld_captions_balanced.jsonl

# scripts/batch_captioning.py
import os
import subprocess
import json
import pandas as pd
from glob import glob
import torchaudio
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
# from utils.audio_utils import whisper_transcribe, analyze_loudness
# from utils.vision_utils import extract_middle_frame
# from utils.caption_utils import caption_image
# from utils.integration_utils import parse_au_intensity, merge_modalities
from utils.utils import extract_middle_frame, caption_image, whisper_transcribe, analyze_loudness, parse_au_intensity, merge_modalities


# === CONFIGURATION ===
OPENFACE_PATH = "../../external/OpenFace/build/bin/FeatureExtraction"
INPUT_FOLDER = "../data/MELD_test_subset/test_oneVideo"
OUTPUT_FOLDER = "../outputs"
AUDIO_TEMP_PATH = "temp_audio.wav"
FRAME_TEMP_PATH = "temp_frame.jpg"
FPS = 30

# Initialize Qwen-Audio
print("[🔄] Loading Qwen-Audio...")
# qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", device_map="auto", trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True).to(device)
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)

def extract_au(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = [OPENFACE_PATH, "-f", video_path, "-aus", "-out_dir", output_dir]
    subprocess.run(command, check=True)

def find_peak_frame(csv_path):
    df = pd.read_csv(csv_path)
    au_presence = [col for col in df.columns if "_c" in col]
    top_aus = df[au_presence].sum().sort_values(ascending=False).head(3).index.str.replace("_c", "")
    top_intensity_cols = [f"{au}_r" for au in top_aus]
    df["emotion_sum"] = df[top_intensity_cols].sum(axis=1)
    peak_idx = df["emotion_sum"].idxmax()
    peak_time = peak_idx / FPS
    return df, peak_idx, peak_time

def extract_audio_segment(video_path, timestamp, duration, output_path):
    start = max(0, timestamp - duration / 2)
    command = [
        "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
        "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def describe_audio_qwen(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    # inputs = qwen_processor(audios=waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(qwen_model.device)
    inputs = qwen_processor(audios=waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
    prompt = "Describe the speaker’s emotional tone, voice intensity, speech style, and delivery. Include observations about pitch, pacing, loudness, hesitation, and clarity. Avoid guessing the speaker's intent or emotion; focus only on vocal characteristics."
    output_ids = qwen_model.generate(**inputs, prompt=prompt, max_new_tokens=100)
    return qwen_processor.batch_decode(output_ids, skip_special_tokens=True)[0]

def process_video(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.csv")
    json_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.json")

    extract_au(video_path, OUTPUT_FOLDER)
    df, peak_idx, peak_sec = find_peak_frame(csv_path)
    au_phrases, raw_aus = parse_au_intensity(csv_path, peak_idx)
    extract_audio_segment(video_path, peak_sec, 2.0, AUDIO_TEMP_PATH)
    audio_desc = describe_audio_qwen(AUDIO_TEMP_PATH)
    transcript = whisper_transcribe(video_path)
    loudness = analyze_loudness(AUDIO_TEMP_PATH)
    extract_middle_frame(video_path, FRAME_TEMP_PATH)
    visual_caption = caption_image(FRAME_TEMP_PATH)
    merged_summary = merge_modalities(au_phrases, f"{loudness}, {audio_desc}", transcript)

    result = {
        "video_id": video_id,
        "peak_time_sec": round(peak_sec, 3),
        "facial_description": au_phrases,
        "audio_description": audio_desc,
        "loudness": loudness,
        "whisper_transcript": transcript,
        "visual_caption": visual_caption,
        "merged_summary": merged_summary,
        "raw_AU_values_at_peak": raw_aus
    }

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[✔] Saved: {json_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    video_files = glob(os.path.join(INPUT_FOLDER, "*.mp4"))
    for video_file in video_files:
        try:
            print(f"▶ Processing: {os.path.basename(video_file)}")
            process_video(video_file)
        except Exception as e:
            print(f"[ERROR] {video_file}: {e}")