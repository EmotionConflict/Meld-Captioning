# Utility functions: frame extraction, Whisper transcription, BLIP captioning
# scripts/utils.py
# AU intensity phrases ("strongly furrows brow"), Audio loudness phrases ("speaks very loudly"), Merged final description without emotions

import cv2
from PIL import Image
import torch
import torchaudio
import pandas as pd
import whisper
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models globally
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
whisper_model = whisper.load_model("small")

# AU Intensity Mapping
def map_au_intensity(value):
    if value < 0.2:
        return "barely"
    elif value < 1.0:
        return "slightly"
    elif value < 2.5:
        return "moderately"
    elif value < 5.0:
        return "strongly"
    else:
        return "very strongly"

# Loudness Mapping
def map_loudness(rms):
    if rms < 0.01:
        return "very softly"
    elif rms < 0.03:
        return "softly"
    elif rms < 0.06:
        return "normally"
    elif rms < 0.1:
        return "loudly"
    else:
        return "very loudly"

# AU to facial phrase mapping
AU_PHRASES = {
    "AU01": "raises the inner eyebrows",
    "AU02": "raises the outer eyebrows",
    "AU04": "furrows the brow",
    "AU07": "tightens the eyelids",
    "AU12": "smiles with mouth corners pulled",
    "AU15": "lowers the mouth corners",
    "AU17": "tightens the chin",
    "AU25": "opens the lips",
    "AU26": "drops the jaw"
}

def extract_middle_frame(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
    cap.release()

def caption_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def whisper_transcribe(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def analyze_loudness(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    rms = waveform.pow(2).mean().sqrt().item()
    return map_loudness(rms)

def parse_au_intensity(openface_csv_path, peak_index):
    df = pd.read_csv(openface_csv_path)
    row = df.iloc[peak_index]
    au_phrases = []
    peak_aus = []

    for au in AU_PHRASES.keys():
        if f"{au}_r" in row:
            value = row[f"{au}_r"]
            if value > 0.1:  # AU activated
                intensity = map_au_intensity(value)
                phrase = AU_PHRASES[au]
                full_phrase = f"{intensity} {phrase}"
                au_phrases.append(full_phrase)
                peak_aus.append(au)

    return au_phrases, peak_aus

def merge_modalities(visual_phrases, audio_phrase, transcript):
    visual_text = " ".join(visual_phrases)
    return (
        f"Visual Description: {visual_text}\n"
        f"Audio Description: {audio_phrase}\n"
        f"Transcript: \"{transcript}\"\n"
        f"Describe what is happening objectively based on these clues."
    )
