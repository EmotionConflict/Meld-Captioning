import torchaudio
import whisper

# Load model once
whisper_model = whisper.load_model("small")

def whisper_transcribe(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def analyze_loudness(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    rms = waveform.pow(2).mean().sqrt().item()
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
