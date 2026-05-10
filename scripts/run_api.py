"""
Script chạy API server + Gradio UI
Chạy: python scripts/run_api.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
import nest_asyncio
import gradio as gr
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

nest_asyncio.apply()

# Global model
model = None
processor = None
MODEL_PATH = "MinhQuy24/Qwen-2B-ViVQA"  # Hoặc "models/student" nếu đã train


def load_vqa_model():
    """Load VQA model khi khởi động"""
    global model, processor
    print("Loading VQA model...")

    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        load_in_4bit=True
    )
    processor = Qwen2VLProcessor.from_pretrained(MODEL_PATH)
    print("Model loaded!")


def get_answer(image_path, question, max_new_tokens=48):
    """Inference function"""
    global model, processor

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        images=[image_path],
        text=[prompt],
        return_tensors="pt"
    ).to(model.device)

    import torch
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# --- FastAPI App ---
app = FastAPI(title="Vi-VQA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_api(image: UploadFile = File(...), question: str = Form(...)):
    """API endpoint cho VQA prediction"""
    import io
    from PIL import Image

    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Lưu tạm ảnh
        temp_path = "temp_image.jpg"
        pil_image.save(temp_path)

        answer = get_answer(temp_path, question)

        # Xóa file tạm
        os.remove(temp_path)

        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
async def root():
    return {"message": "Vi-VQA API", "docs": "/docs"}


def run_server():
    """Chạy FastAPI server"""
    uvicorn.run(app, host="127.0.0.1", port=8000)


# --- Gradio UI ---
def vqa_ui_wrapper(image_file, user_question):
    """Wrapper để gọi API từ Gradio"""
    if image_file is None or not user_question:
        return "Vui lòng cung cấp ảnh và câu hỏi."

    try:
        url = "http://127.0.0.1:8000/predict"
        with open(image_file, "rb") as f:
            files = {"image": f}
            data = {"question": user_question}
            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("answer", res_json.get("error", "Lỗi không xác định"))
        else:
            return f"Lỗi server: {response.status_code}"
    except Exception as e:
        return f"Không thể kết nối server: {str(e)}"


def launch_ui():
    """Launch Gradio UI"""
    with gr.Blocks() as demo:
        gr.Markdown("## Vietnamese Visual Question Answering (Vi-VQA)")
        gr.Markdown("Knowledge Distillation: Qwen2-VL-7B → Qwen2-VL-2B")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type='filepath', label='Upload Image')
                input_txt = gr.Textbox(
                    lines=2,
                    label='Question',
                    placeholder='vd: Mô tả bức ảnh này'
                )
                btn = gr.Button('Submit', variant='primary')

            with gr.Column():
                output_txt = gr.Textbox(label='Answer', interactive=False)

        btn.click(fn=vqa_ui_wrapper, inputs=[input_img, input_txt], outputs=output_txt)

    demo.launch(share=False, debug=True)


def main():
    print("="*50)
    print("Vi-VQA API Server + Gradio UI")
    print("="*50)

    # Load model
    load_vqa_model()

    # Chạy server trong thread riêng
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("API Server started at http://127.0.0.1:8000")
    print("API Docs at http://127.0.0.1:8000/docs")
    print()

    # Launch Gradio UI
    print("Launching Gradio UI...")
    launch_ui()


if __name__ == "__main__":
    main()