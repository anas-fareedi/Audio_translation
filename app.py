# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from pathlib import Path
# from google import genai

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# genai.configure(api_key=GOOGLE_API_KEY)

# app = FastAPI()
# UPLOAD_FOLDER = "static"

# # Create upload folder if it doesn't exist
# Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# # Mount static files
# app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")


# @app.get('/')
# async def get_main():
#     with open("templates/index.html", "r") as f:
#         return f.read()


# @app.post('/')
# async def post_main(file: UploadFile = File(...), language: str = Form(...)):
#     if file:
#         filename = file.filename
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
        
#         # Save uploaded file
#         with open(file_path, "wb") as f:
#             contents = await file.read()
#             f.write(contents)
        
#         # Translate audio
#         with open(file_path, "rb") as audio_file:
#             transcript = genai.audio.translate("whisper-1", audio_file)

#         # Translate text to target language
#         response = genai.chat.completions.create(
#                 model="gpt-4",
#                 messages = [{ "role": "system", "content": f"You will be provided with a sentence in English, and your task is to translate it into {language}" }, 
#                             { "role": "user", "content": transcript.text }],
#                 temperature=0,
#                 max_tokens=256
#               )
        
#         return JSONResponse(content=response)
    
#     return JSONResponse(content={"error": "No file provided"}, status_code=400)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)

import os
import shutil
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_MODEL")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load Whisper model
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
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = asr(file_path)

    return {
        "transcription": result["text"]
    }
