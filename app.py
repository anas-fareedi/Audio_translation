import os
import shutil
from fastapi import FastAPI, File, UploadFile, Request ,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
HF_TOKEN = os.getenv("HF_MODEL")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

asr = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base",
    token=HF_TOKEN
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    generate_kwargs = {
        "task": "transcribe"
    }
    if language != "auto" and language != "hi":
        generate_kwargs["language"] = language

    result = asr(file_path, generate_kwargs=generate_kwargs)

    return {
        "selected_language": language,
        "transcription": result["text"]
    }