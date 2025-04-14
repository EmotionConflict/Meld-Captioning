import os
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm

# ====== Settings ======

meld_root = './data/MELD.Raw'
test_csv_path = os.path.join(meld_root, 'test_sent_emo.csv')
output_repeated_dir = os.path.join(meld_root, 'output_repeated_splits_test')

output_dir = './data/MELD_test_subset'
subset_video_audio_dir = os.path.join(output_dir, 'test_subset')
subset_subtitle_dir = os.path.join(output_dir, 'test_subtitles')
subset_audio_dir = os.path.join(output_dir, 'test_subset_wav')

samples_per_emotion = 50
temporal_window = 1

# ====== Create Output Folders ======

os.makedirs(subset_video_audio_dir, exist_ok=True)
os.makedirs(subset_subtitle_dir, exist_ok=True)
os.makedirs(subset_audio_dir, exist_ok=True)

# ====== Load Metadata ======

test_labels = pd.read_csv(test_csv_path)
print(f"Loaded {len(test_labels)} total test samples.")

# ====== Balanced Sampling ======

subset_labels_core = test_labels.groupby('Emotion', group_keys=False).sample(n=samples_per_emotion, random_state=42)
print(f"Sampled {len(subset_labels_core)} core samples (balanced across emotions).")

# ====== Include Temporal Neighbors ======

test_labels['Dialogue_Utt'] = test_labels['Dialogue_ID'].astype(str) + '_' + test_labels['Utterance_ID'].astype(str)
test_labels.set_index('Dialogue_Utt', inplace=True)

all_selected_indices = set()

for idx, row in subset_labels_core.iterrows():
    dialogue_id = row['Dialogue_ID']
    utterance_id = row['Utterance_ID']
    
    all_selected_indices.add(f"{dialogue_id}_{utterance_id}")
    
    for offset in range(-temporal_window, temporal_window + 1):
        if offset == 0:
            continue
        neighbor_id = utterance_id + offset
        neighbor_key = f"{dialogue_id}_{neighbor_id}"
        if neighbor_key in test_labels.index:
            all_selected_indices.add(neighbor_key)

print(f"After adding temporal neighbors: {len(all_selected_indices)} utterances total.")

# ====== Create Subset Metadata ======

subset_labels_full = test_labels.loc[list(all_selected_indices)].reset_index()
subset_labels_full[['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Utterance', 'Emotion']].to_csv(
    os.path.join(output_dir, 'test_labels_subset.csv'), index=False)
print(f"Saved full subset metadata to {output_dir}/test_labels_subset.csv.")

# ====== Copy Video Files and Generate Subtitles ======

for idx, row in tqdm(subset_labels_full.iterrows(), total=len(subset_labels_full), desc="Copying video and subtitle files"):
    utt_id = row['Utterance_ID']
    dia_id = row['Dialogue_ID']
    utterance = str(row['Utterance'])

    base_name = f"dia{dia_id}_utt{utt_id}"

    mp4_src = os.path.join(output_repeated_dir, f"{base_name}.mp4")
    mp4_dst = os.path.join(subset_video_audio_dir, f"{base_name}.mp4")
    
    if os.path.exists(mp4_src):
        shutil.copy(mp4_src, mp4_dst)
    else:
        print(f"Warning: Missing {mp4_src}")

    subtitle_filename = os.path.join(subset_subtitle_dir, f"{base_name}.txt")
    with open(subtitle_filename, 'w', encoding='utf-8') as f:
        f.write(utterance)

print("Video copying and subtitle generation complete.")

# ====== Function: Extract .wav from .mp4 ======

def extract_wav_from_mp4(mp4_folder, wav_folder, sample_rate=16000):
    os.makedirs(wav_folder, exist_ok=True)

    mp4_files = [f for f in os.listdir(mp4_folder) if f.endswith('.mp4')]

    print(f"Found {len(mp4_files)} .mp4 files in {mp4_folder}.")

    for mp4_file in tqdm(mp4_files, desc="Extracting audio"):
        mp4_path = os.path.join(mp4_folder, mp4_file)
        wav_filename = os.path.splitext(mp4_file)[0] + '.wav'
        wav_path = os.path.join(wav_folder, wav_filename)

        command = [
            'ffmpeg',
            '-y',
            '-i', mp4_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            wav_path
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Extraction complete. Saved .wav files in {wav_folder}.")

# ====== Run Audio Extraction ======

extract_wav_from_mp4(subset_video_audio_dir, subset_audio_dir)

print("\nSubset and audio extraction complete.")
print(f"Videos saved in: {subset_video_audio_dir}")
print(f"Audios saved in: {subset_audio_dir}")
print(f"Subtitles saved in: {subset_subtitle_dir}")
print(f"Metadata saved in: {output_dir}/test_labels_subset.csv")
