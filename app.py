import gradio as gr
from nllb import translation, NLLB_EXAMPLES
from flores200_codes import flores_codes

lang_codes = list(flores_codes.keys())

nllb_translate = gr.Interface(
    fn=translation,
    inputs=[
        gr.Dropdown(
            ["nllb-1.3B", "nllb-distilled-1.3B", "nllb-3.3B"],
            label="Model",
            value="nllb-distilled-1.3B",
        ),
        gr.Dropdown(
            lang_codes,
            label="Source language",
            value="English",
        ),
        gr.Dropdown(
            lang_codes,
            label="Target language",
            value="Shan",
        ),
        gr.Textbox(lines=5, label="Input text"),
    ],
    outputs="json",
    examples=NLLB_EXAMPLES,
    title="NLLB Translation Demo",
    description="Translate text from one language to another.",
    allow_flagging="never",
)

with gr.Blocks() as demo:
    nllb_translate.render()

demo.launch()
