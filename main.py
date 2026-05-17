import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import io
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

app = FastAPI()

# ===== CONFIG =====
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen2.5-VL-3B-Instruct")

# ===== LOAD MODEL ONCE =====
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ===== LOAD SYSTEM PROMPTS =====
def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

system_ocr_en = load_prompt("system_prompts/ocr_eng_system_prompt.txt")
system_ocr_zh = load_prompt("system_prompts/ocr_ch_system_prompt.txt")
system_txt_en = load_prompt("system_prompts/txt_eng_system_prompt.txt")
system_txt_zh = load_prompt("system_prompts/txt_ch_system_prompt.txt")


# ===== CORE MODEL FUNCTION =====
def run_model(messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


# ===== OCR ENDPOINT =====
@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form("en")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_ocr_en if lang == "en" else system_ocr_zh}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}
                ],
            }
        ]

        result = run_model(messages)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== TEXT FIX ENDPOINT =====
@app.post("/fix")
async def fix(
    text: str = Form(...),
    lang: str = Form("en")
):
    try:
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_txt_en if lang == "en" else system_txt_zh}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ],
            }
        ]

        result = run_model(messages)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))