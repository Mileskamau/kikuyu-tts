import os
import io
import torch
import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer

# Create FastAPI app
app = FastAPI()

# Global variables to cache model (persists across requests)
_model = None
_tokenizer = None

def load_model():
    """Lazy load the model on first request"""
    global _model, _tokenizer
    if _model is None:
        print("Loading Kikuyu TTS model from Hugging Face...")
        _model = VitsModel.from_pretrained("facebook/mms-tts-kik")
        _tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kik")
        print("Model loaded successfully")
    return _model, _tokenizer

class TTSRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Kikuyu TTS API is running", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/tts")
async def tts(request: TTSRequest):
    try:
        # Load model (first request will be slow)
        model, tokenizer = load_model()
        
        # Generate audio
        inputs = tokenizer(request.text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Convert to WAV format
        audio_np = output.squeeze().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype('int16')
        
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, rate=model.config.sampling_rate, data=audio_int16)
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
