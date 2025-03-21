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

# Check if model exists locally
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
else:
    print("Model already exists locally.")

# Load model into memory
model = joblib.load(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

def extract_features(audio, sr):
    """Extract features from audio."""
    # Your feature extraction code here...

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
