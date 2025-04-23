import os
import subprocess
import pandas as pd
import torchaudio
import torch
import json
from glob import glob
from transformers import AutoProcessor, AutoModelForCausalLM
import whisper

# === CONFIG ===
OPENFACE_PATH = "../../external/OpenFace/build/bin/FeatureExtraction"  # assuming OpenFace is cloned under external/
INPUT_FOLDER = "../data"                  # where your MELD .mp4 videos are
OUTPUT_FOLDER = "../outputs"              # where JSON results will be stored
AUDIO_TEMP_PATH = "temp_audio.wav"        # local to the script folder

AUDIO_TEMP_PATH = "temp_audio.wav"
FPS = 30
WHISPER_MODEL_NAME = "medium"

AU_PHRASES = {
    'AU01': 'Inner Brow Raiser', 'AU02': 'Outer Brow Raiser', 'AU04': 'Brow Lowerer',
    'AU05': 'Upper Lid Raiser', 'AU06': 'Cheek Raiser', 'AU07': 'Lid Tightener',
    'AU09': 'Nose Wrinkler', 'AU10': 'Upper Lip Raiser', 'AU12': 'Lip Corner Puller',
    'AU14': 'Dimpler', 'AU15': 'Lip Corner Depressor', 'AU17': 'Chin Raiser',
    'AU20': 'Lip stretcher', 'AU23': 'Lip Tightener', 'AU25': 'Lips Part',
    'AU26': 'Jaw Drop', 'AU28': 'Lip Suck', 'AU45': 'Blink'
}

def map_au_intensity(value):
    if value < 0.2: return "barely"
    elif value < 1.0: return "slightly"
    elif value < 2.5: return "moderately"
    elif value < 5.0: return "strongly"
    else: return "very strongly"

def extract_au(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = [OPENFACE_PATH, "-f", video_path, "-aus", "-out_dir", output_dir]
    subprocess.run(command, check=True)

def find_peak(csv_path):
    df = pd.read_csv(csv_path)
    au_presence = [col for col in df.columns if "_c" in col]
    top_aus = df[au_presence].sum().sort_values(ascending=False).head(3).index.str.replace("_c", "")
    intensity_cols = [f"{au}_r" for au in top_aus]
    df["emotion_sum"] = df[intensity_cols].sum(axis=1)
    idx = df["emotion_sum"].idxmax()
    time = idx / FPS
    return df, idx, time

def get_au_description(df, idx):
    row = df.iloc[min(idx, len(df)-1)]
    phrases, raw_aus = [], {}
    for au, label in AU_PHRASES.items():
        col = f"{au}_r"
        if col in row and row[col] > 0.1:
            phrases.append(f"{map_au_intensity(row[col])} {label}")
            raw_aus[au] = float(row[col])
    return phrases, raw_aus

def extract_audio(video_path, timestamp, duration, output_path):
    start = max(0, timestamp - duration / 2)
    command = ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
               "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", output_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def describe_audio_qwen(audio_path, model, processor):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    inputs = processor(audios=waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(model.device)
    prompt = "Describe the speaker’s emotional tone, speech style, and delivery."
    outputs = model.generate(**inputs, prompt=prompt, max_new_tokens=100)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

def transcribe_whisper(audio_path, whisper_model):
    result = whisper_model.transcribe(audio_path)
    return result.get("text", "").strip()

def process_video(video_path, qwen_model, qwen_processor, whisper_model):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.csv")
    json_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.json")
    extract_au(video_path, OUTPUT_FOLDER)
    df, peak_idx, peak_sec = find_peak(csv_path)
    au_desc, au_vals = get_au_description(df, peak_idx)
    extract_audio(video_path, peak_sec, 2.0, AUDIO_TEMP_PATH)
    audio_desc = describe_audio_qwen(AUDIO_TEMP_PATH, qwen_model, qwen_processor)
    transcript = transcribe_whisper(video_path, whisper_model)
    result = {
        "video_id": video_id,
        "peak_time_sec": round(peak_sec, 3),
        "facial_description": au_desc,
        "audio_description": audio_desc,
        "whisper_transcript": transcript,
        "raw_AU_values_at_peak": au_vals
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

# if __name__ == "__main__":
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", device_map="auto", trust_remote_code=True)
#     qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)
#     whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
#     for video_file in glob(os.path.join(INPUT_FOLDER, "*.mp4")):
#         try:
#             process_video(video_file, qwen_model, qwen_processor, whisper_model)
#         except Exception as e:
#             print(f"[ERROR] {video_file}: {e}")