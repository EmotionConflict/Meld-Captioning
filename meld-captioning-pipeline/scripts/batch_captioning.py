# Batch captioning script for MELD dataset
# Load each video/audio/subtitle
# Extract middle frame
# Run BLIP-2 for frame caption
# Run Whisper for audio transcription
# Merge everything into one paragraph
# Save into meld_captions_balanced.jsonl

# scripts/batch_captioning.py

# This batch script processes: MELD video/audio/subtitle, OpenFace outputs (AU parsing), Merges into richer JSON structure per sample.



from utils import (
    extract_middle_frame, caption_image, whisper_transcribe,
    analyze_loudness, parse_au_intensity, merge_modalities
)
import pandas as pd
import os
import json
from tqdm import tqdm

def batch_caption(df, video_dir, audio_dir, subtitle_dir, openface_dir, output_path):
    results = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        utt_id = row['Utterance_ID']
        dia_id = row['Dialogue_ID']
        
        base_name = f"dia{dia_id}_utt{utt_id}"
        video_path = os.path.join(video_dir, f"{base_name}.mp4")
        audio_path = os.path.join(audio_dir, f"{base_name}.wav")
        subtitle_path = os.path.join(subtitle_dir, f"{base_name}.txt")
        openface_csv_path = os.path.join(openface_dir, f"{base_name}.csv")
        frame_path = f"frame_{base_name}.jpg"
        
        extract_middle_frame(video_path, frame_path)
        visual_caption = caption_image(frame_path)
        audio_text = whisper_transcribe(audio_path)
        audio_loudness = analyze_loudness(audio_path)
        
        transcript = open(subtitle_path, 'r', encoding='utf-8').read().strip()
        
        # Find peak AU frame
        df_openface = pd.read_csv(openface_csv_path)
        peak_index = df_openface['frame'].idxmax()
        
        visual_prior_list, peak_AU_list = parse_au_intensity(openface_csv_path, peak_index)
        
        merged_caption = merge_modalities(visual_prior_list, audio_loudness, transcript)
        
        sample_id = f"sample_{idx:08d}"
        results[sample_id] = {
            "AU_list": list(df_openface.columns[df_openface.columns.str.contains('AU')]),
            "visual_prior_list": visual_prior_list,
            "audio_prior_list": audio_loudness,
            "peak_index": int(peak_index),
            "peak_AU_list": peak_AU_list,
            "text": transcript,
            "smp_reason_caption": merged_caption
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} samples to {output_path}")
