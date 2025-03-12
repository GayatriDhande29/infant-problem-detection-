import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "dataset/"

# Load audio
def load_audio(file_path, target_sr=22050):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

# Convert to spectrogram
def extract_spectrogram(audio, sr):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    return librosa.power_to_db(spectrogram, ref=np.max)

# Extract MFCCs
def extract_mfcc(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T

# Normalize
def normalize_features(features):
    return (features - np.mean(features)) / np.std(features)

# Process dataset
def process_dataset(dataset_path):
    data = []
    for label, category in enumerate(["cry", "not_cry"]):
        folder_path = os.path.join(dataset_path, category)
        
        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {category}"):
            file_path = os.path.join(folder_path, file_name)
            
            audio, sr = load_audio(file_path)
            spectrogram = extract_spectrogram(audio, sr)
            mfccs = extract_mfcc(audio, sr)
            
            spectrogram = normalize_features(spectrogram)
            mfccs = normalize_features(mfccs)
            
            data.append([spectrogram, mfccs, label])
    
    return data

# Run preprocessing and save data
if __name__ == "__main__":
    dataset = process_dataset(DATASET_PATH)
    df = pd.DataFrame(dataset, columns=["spectrogram", "mfccs", "label"])
    df.to_pickle("preprocessed_data.pkl")
    print("✅ Preprocessing Done! Saved as 'preprocessed_data.pkl'")
