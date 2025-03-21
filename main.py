import librosa
import numpy as np
import joblib
import requests
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
import os

# Hugging Face model details
HF_USERNAME = "udaysharma123"
HF_MODEL_REPO = "colon_final"
MODEL_FILENAME = "best_rf_model.pkl"
MODEL_PATH = f"./{MODEL_FILENAME}"

# Function to download model from Hugging Face if not found locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Hugging Face...")
        url = f"https://huggingface.co/{HF_USERNAME}/{HF_MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

# Initialize FastAPI app
app = FastAPI()

# Download model at startup
download_model()

# Load the model once during startup
model = joblib.load(MODEL_PATH)

def extract_features(audio, sr):
    """Extract features from audio."""
    pitch_values = librosa.yin(audio, fmin=50, fmax=300)
    pitch_mean, pitch_std, pitch_range = np.mean(pitch_values), np.std(pitch_values), np.ptp(pitch_values)
    
    rms_energy = librosa.feature.rms(y=audio).flatten()
    intensity_mean, intensity_std, intensity_range = np.mean(rms_energy), np.std(rms_energy), np.ptp(rms_energy)
    
    duration = librosa.get_duration(y=audio, sr=sr)
    peaks = librosa.effects.split(audio, top_db=30)
    speech_rate = len(peaks) / duration if duration > 0 else 0
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    feature_vector = np.hstack([
        pitch_mean, pitch_std, pitch_range,
        intensity_mean, intensity_std, intensity_range,
        speech_rate, spectral_centroid, spectral_rolloff, zcr,
        mfccs_mean
    ])
    return feature_vector

def predict_emotion(audio, sr):
    """Predict emotion from extracted features."""
    features = extract_features(audio, sr)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint to accept a .wav file and return predicted mood."""
    try:
        audio_bytes = await file.read()
        audio_buffer = BytesIO(audio_bytes)
        
        audio, sr = librosa.load(audio_buffer, sr=22050)
        mood = predict_emotion(audio, sr)
        
        return {"mood": mood}
    except Exception as e:
        return {"error": str(e)}

# Run locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
