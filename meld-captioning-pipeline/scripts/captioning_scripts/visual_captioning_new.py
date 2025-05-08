import sys
import os
import io
import json
import traceback
import cv2
from PIL import Image
import torch
import pandas as pd
# import torchaudio
# from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, pipeline
import base64

# === Path Setup ===
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AU_labeling.au_extraction_new import map_au_intensity, extract_au_from_video, parse_au_intensity
from AU_labeling.peak_frame_description_new import find_peak_frame, extract_frame_by_index
from process_meld_multimodal import describe_audio_qwen

# === Output Path ===
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MELD_test_subset/MELD_annotations.json"))
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === OpenAI GPT-4o client ===
from openai import OpenAI
client = OpenAI(api_key="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") #API-KEY
#wrongclient = OpenAI(api_key="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")


# client = OpenAI(api_key="API-KEY") 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# blip_model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-flan-t5-xl",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
# ).to(device)

# qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True).to(device)
# qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# llama_model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3-8B",
#     torch_dtype="auto"  # Optimized dtype for the hardware
# ).to(device)
# # Create pipeline
# llama = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)

# === AU to Facial Phrase Mapping ===
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

# === Input Video Directory ===
directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MELD_test_subset/test_subset"))
if not os.path.exists(directory_path):
    print(f"[❌] Directory does not exist: {directory_path}")
else:
    print(f"[✅] Directory exists. Continuing...")

# === Main Processing Loop ===
total=0
for root, dirs, files in os.walk(directory_path):
    for file in files:
        # print(f"[▶] Processing file: {file}")
        file_path = os.path.join(root, file)
        # print(f"Found file: {file_path}")
        video_id= file

        # === AU Extraction ===
        os.makedirs("AU_labeling/AU_data", exist_ok=True)
        # extract_au_from_video(file_path, "AU_labeling/AU_data", "../OpenFace/build/bin/FeatureExtraction")
        openface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../OpenFace/build/bin/FeatureExtraction"))

        # print(f"[INFO] Extracting AU for: {video_id}")
        extract_au_from_video(file_path, "AU_labeling/AU_data", openface_path)

        # print("[DEBUG] AU extraction done")

        peak_frame, time= find_peak_frame(f"AU_labeling/AU_data/{file[:-3]}csv")
        au_phrases, raw_aus = parse_au_intensity(f"AU_labeling/AU_data/{file[:-3]}csv", peak_frame)

        # === Visual Frame Description ===
        # print("started image description")
        prompt = "Describe the setting and what the people in the frame are doing."
        image = extract_frame_by_index(file_path, peak_frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")  # or "PNG"
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

        prompt = "Describe what is happening in this video frame as if you're narrating it to someone who cannot see it. Focus only on visible details such as people’s actions, facial expressions, gestures, body language, clothing, objects, and the physical setting. Be specific about how people are positioned and how they interact with each other and their surroundings. Write descriptively—do not simply list objects. Include visual cues that might suggest emotional states, but don’t speculate beyond what’s visible."
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Must be vision-capable
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        visual_caption=response.choices[0].message.content
        # inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16 if torch.cuda.is_available() else torch.float32)
        # generated_ids = blip_model.generate(**inputs, max_new_tokens=20)
        # # print("Generated IDs:", generated_ids)
        # visual_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # # print("Decoded:", visual_caption)

        # print("started audio description")
        # waveform, sr = torchaudio.load(f"data/MELD_test_subset/test_subset_wav/{file[:-3]}wav")
        # if sr != 16000:
        #     waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        # inputs = qwen_processor(text="Describe the speaker’s emotional tone, speech style, and delivery.",audios=waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
        # print("generating output")
        # outputs = qwen_model.generate(**inputs, max_new_tokens=20)
        # print("decoding output")
        # audio_description = qwen_processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]
        # print(audio_description)

        subtitle=""
        subtitle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../data/MELD_test_subset/test_subtitles/{file[:-3]}txt"))
        with open(subtitle_path, "r") as f: #f"../data/MELD_test_subset/test_subtitles/{file[:-3]}txt"
            subtitle = f.read()

        coarse_summary=f"{visual_caption} The facial expressions include {', '.join(au_phrases)}. Saying: '{subtitle}'."
        

        # print("started summary generation")
        # # Generate text
        # prompt = f"System: You are an emotion analysis expert. Please infer emotion label based on the given the emotional features.\nQuestion: {coarse_summary}. Please sort out the correct emotional clues and infer why the person in the video feels that emotion."
        # print(prompt)
        # fine_summary = llama(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        # fine_summary = fine_summary[len(prompt):].strip()
        # print(fine_summary)
        

        result = {
            "video_id": video_id,
            # "AU_raw_intensities": au_intensities,
            "visual_expression_description": au_phrases,
            "visual_objective_description": visual_caption,
            "peak_time": time,
            "raw_AU_values_at_peak": raw_aus,
            "coarse-grained_summary": coarse_summary,
            # "fine-grained_summary": fine_summary,
        }

        # print(result)
        os.makedirs("../data/MELD_test_subset", exist_ok=True)
        try:
            with open(f"../data/MELD_test_subset/MELD_annotations.json", "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []

        # === Append to JSON Output ===
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                data = json.load(f)
        else:
            data = [] #new block

        data.append(result)
        # print(data)

        # with open(f"../data/MELD_test_subset/MELD_annotations.json", "w") as f:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[✔] Saved to: ../data/MELD_test_subset/MELD_annotations.json")
        total+=1
        print(total)
        # break
