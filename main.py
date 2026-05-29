import os
import base64
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from openai import OpenAI

app = FastAPI()

LLAMA_API_BASE = os.getenv("LLAMA_API_BASE", "http://localhost:8080/v1")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "local-model")

client = OpenAI(base_url=LLAMA_API_BASE, api_key=LLAMA_API_KEY)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PROMPTS = {
    "ocr": {
        "en": load_prompt("system_prompts/ocr_eng_system_prompt.txt"),
        "zh": load_prompt("system_prompts/ocr_ch_system_prompt.txt"),
    },
    "txt": {
        "en": load_prompt("system_prompts/txt_eng_system_prompt.txt"),
        "zh": load_prompt("system_prompts/txt_ch_system_prompt.txt"),
    },
    "emotion": load_prompt("system_prompts/emotion_system_prompt.txt"),
}


def get_prompt(task: str, lang: str) -> str:
    prompt = PROMPTS[task].get(lang)
    if not prompt:
        raise HTTPException(status_code=400, detail="Invalid language")
    return prompt


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def run_model(messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM API error: {e}") from e


def detect_emotion(text: str) -> str:
    return run_model([
        {"role": "system", "content": PROMPTS["emotion"]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ])


def format_response(result: str, skip_emotion: bool) -> dict:
    return {
        "result": result,
        "emotion": None if skip_emotion else detect_emotion(result),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    skip_emotion: bool = Form(True),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_base64 = image_to_base64(image)

        messages = [
            {"role": "system", "content": get_prompt("ocr", lang)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            },
        ]
        return format_response(run_model(messages), skip_emotion)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/fix")
async def fix(
    text: str = Form(...),
    lang: str = Form("en"),
    skip_emotion: bool = Form(True),
):
    try:
        messages = [
            {"role": "system", "content": get_prompt("txt", lang)},
            {"role": "user", "content": text},
        ]
        return format_response(run_model(messages), skip_emotion)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
