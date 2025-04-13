# Batch captioning script for MELD dataset
# Load each video/audio/subtitle
# Extract middle frame
# Run BLIP-2 for frame caption
# Run Whisper for audio transcription
# Merge everything into one paragraph
# Save into meld_captions_balanced.jsonl

from utils import extract_middle_frame, caption_image, whisper_transcribe, extract_subtitle_text, merge_modalities
import os
import pandas as pd
import json
from tqdm import tqdm

def batch_caption(df, video_dir, audio_dir, subtitle_dir, output_path):
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        utt_id = row['Utterance_ID']
        dia_id = row['Dialogue_ID']
        emotion = row['Emotion']
        
        video_path = os.path.join(video_dir, f"dia{dia_id}_utt{utt_id}.mp4")
        audio_path = os.path.join(audio_dir, f"dia{dia_id}_utt{utt_id}.wav")
        subtitle_path = os.path.join(subtitle_dir, f"dia{dia_id}_utt{utt_id}.txt")
        frame_path = f"frame_{dia_id}_{utt_id}.jpg"
        
        extract_middle_frame(video_path, frame_path)
        visual_caption = caption_image(frame_path)
        audio_text = whisper_transcribe(audio_path)
        subtitle_text = extract_subtitle_text(subtitle_path)
        merged_text = merge_modalities(audio_text, visual_caption, subtitle_text)
        
        results.append({
            "id": f"dia{dia_id}_utt{utt_id}",
            "multimodal_description": merged_text,
            "emotion_label": emotion
        })
        
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')

    print(f"Saved {len(results)} samples to {output_path}")
