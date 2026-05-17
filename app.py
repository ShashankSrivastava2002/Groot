"""
FastAPI server wrapping the Gemma 4 ReAct agent.
Endpoints:
  POST /chat       — send message + optional image, get agent response
  POST /transcribe — transcribe uploaded audio (WAV) to text
  POST /reset      — clear conversation history
  GET  /           — serve the chat UI
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import numpy as np

from tool_call_gemma import (
    react_agent,
    MESSAGES,
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
)
from transcription import get_transription

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.post("/chat")
async def chat(
    message: str = Form(""),
    image: UploadFile | None = File(None),
    language: str = Form("English"),
):

    print("got route")

    # return {"response": language}

    images = []
    if image and image.filename:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        images = [pil_image]

    answer = react_agent(
        user_prompt=message,
        images=images if images else None,
        tools=TOOL_SCHEMAS,
        language=language,
    )

    return {"response": answer}


@app.post("/reset")
async def reset():
    print("CALLED RESET")
    # Empty list — system prompt will be re-added with correct language on next chat call
    MESSAGES.clear()
    return {"status": "ok", "message": "Conversation reset."}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Receive WAV audio and return transcription."""
    import wave

    contents = await audio.read()

    # Parse WAV bytes → float32 numpy array
    with wave.open(io.BytesIO(contents), "rb") as wf:
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(n_frames)

    # Convert based on sample width
    if sample_width == 2:
        audio_arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio_arr = np.frombuffer(raw, dtype=np.float32)
    else:
        audio_arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if framerate != 16000:
        ratio = 16000 / framerate
        new_len = int(len(audio_arr) * ratio)
        audio_arr = np.interp(
            np.linspace(0, len(audio_arr), new_len, endpoint=False),
            np.arange(len(audio_arr)),
            audio_arr,
        ).astype(np.float32)

    transcription = get_transription(audio_arr)
    if transcription is None:
        return {"text": "", "error": "Transcription failed"}

    text = transcription[0] if isinstance(transcription, list) else str(transcription)
    return {"text": text.strip()}
