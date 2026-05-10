"""
Gradio Frontend
UI cho Vi-VQA demo
"""
import gradio as gr

def create_vqa_ui(predict_fn):
    """
    Create Gradio UI cho VQA demo

    Args:
        predict_fn: Function nhận (image, question) và trả về answer
    """
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

        btn.click(fn=predict_fn, inputs=[input_img, input_txt], outputs=output_txt)

    return demo

def run_ui(predict_fn, share=True):
    """Launch Gradio UI"""
    demo = create_vqa_ui(predict_fn)
    demo.launch(share=share, debug=True)