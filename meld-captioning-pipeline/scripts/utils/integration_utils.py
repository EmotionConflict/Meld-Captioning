import pandas as pd

AU_PHRASES = {
    "AU01": "raises the inner eyebrows", "AU02": "raises the outer eyebrows",
    "AU04": "furrows the brow", "AU07": "tightens the eyelids",
    "AU12": "smiles with mouth corners pulled", "AU15": "lowers the mouth corners",
    "AU17": "tightens the chin", "AU25": "opens the lips", "AU26": "drops the jaw"
}

def map_au_intensity(value):
    if value < 0.2: return "barely"
    elif value < 1.0: return "slightly"
    elif value < 2.5: return "moderately"
    elif value < 5.0: return "strongly"
    else: return "very strongly"

def parse_au_intensity(openface_csv_path, peak_index):
    df = pd.read_csv(openface_csv_path)
    row = df.iloc[peak_index]
    au_phrases, peak_aus = [], []

    for au in AU_PHRASES.keys():
        if f"{au}_r" in row:
            value = row[f"{au}_r"]
            if value > 0.1:
                intensity = map_au_intensity(value)
                au_phrases.append(f"{intensity} {AU_PHRASES[au]}")
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
