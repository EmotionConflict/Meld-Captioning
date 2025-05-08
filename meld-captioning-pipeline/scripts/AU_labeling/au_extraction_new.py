import os
import subprocess
import argparse
import pandas as pd

# AU to facial phrase mapping
AU_PHRASES = {
    'AU01': 'Inner Brow Raiser',
    'AU02': 'Outer Brow Raiser',
    'AU04': 'Brow Lowerer',
    'AU05': 'Upper Lid Raiser',
    'AU06': 'Cheek Raiser',
    'AU07': 'Lid Tightener',
    'AU09': 'Nose Wrinkler',
    'AU10': 'Upper Lip Raiser',
    'AU12': 'Lip Corner Puller',
    'AU14': 'Dimpler',
    'AU15': 'Lip Corner Depressor',
    'AU17': 'Chin Raiser',
    'AU20': 'Lip stretcher',
    'AU23': 'Lip Tightener',
    'AU25': 'Lips Part',
    'AU26': 'Jaw Drop',
    'AU28': 'Lip Suck',
    'AU45': 'Blink'
}

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

def extract_au_from_video(video_path, output_dir, openface_bin_path):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the command to run OpenFace's FeatureExtraction
    command = [
        openface_bin_path,
        "-f", video_path,
        "-aus",                      # Enable AU detection
        "-out_dir", output_dir       # Where to save the .csv output
    ]

    print(f"Running command:\n{' '.join(command)}\n")
    subprocess.run(command, check=True)
    print(f"AU data saved to: {output_dir}")

def find_peak_frame(au_data_path):
    # === LOAD AU DATA ===
    df = pd.read_csv(au_data_path)

    # Extract AU presence and intensity columns
    au_presence_cols = [col for col in df.columns if "_c" in col]
    au_intensity_cols = [col for col in df.columns if "_r" in col]

    # Step 1: Identify most frequently occurring AUs
    au_frequencies = df[au_presence_cols].sum().sort_values(ascending=False)
    most_frequent_aus = au_frequencies.head(3).index.str.replace("_c", "")

    # Step 2: Sum the intensity values of these AUs for each frame
    relevant_intensity_cols = [au + "_r" for au in most_frequent_aus]
    df["emotion_sum"] = df[relevant_intensity_cols].sum(axis=1)

    # Step 3: Find emotional peak frame
    peak_frame_index = df["emotion_sum"].idxmax()
    return peak_frame_index, peak_frame_index/30


def parse_au_intensity(openface_csv_path, peak_index):
    try:
        df = pd.read_csv(openface_csv_path)
        if peak_index >= len(df):
            peak_index = len(df) // 2  # fallback to middle if peak invalid
        row = df.iloc[peak_index]

        au_phrases = []
        peak_aus = {}

        for au in AU_PHRASES.keys():
            if f"{au}_r" in row:
                value = row[f"{au}_r"]
                if value > 0.1:
                    intensity = map_au_intensity(value)
                    phrase = AU_PHRASES[au]
                    full_phrase = f"{intensity} {phrase}"
                    au_phrases.append(full_phrase)
                peak_aus[au]=value

        return au_phrases, peak_aus
    except Exception as e:
        print(f"[ERROR] AU parsing failed for {openface_csv_path}: {e}")
        return [], []

# if __name__ == "__main__":
#     os.makedirs("../../outputs/AU_labeling", exist_ok=True)
#     extract_au_from_video("../../data/MELD_test_subset/test_subset/dia12_utt12.mp4", "../../outputs/AU_labeling", "../../OpenFace/build/bin/FeatureExtraction")
#     idx= find_peak_frame("../../outputs/AU_labeling/dia12_utt12.csv")
#     print(parse_au_intensity("../../outputs/AU_labeling/dia12_utt12.csv", idx))

# if __name__ == "__main__":
#     os.makedirs("AU_data", exist_ok=True)
#     extract_au_from_video("../../data/MELD_test_subset/test_subset/dia12_utt12.mp4", "AU_data", "../../OpenFace/build/bin/FeatureExtraction")
#     idx= find_peak_frame("AU_data/dia12_utt12.csv")
#     print(parse_au_intensity("AU_data/dia12_utt12.csv", idx))
