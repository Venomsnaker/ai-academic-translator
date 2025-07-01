import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "pigpig1524/ml-translator"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="vi_VN")
model.to(device)

def translate(text, model=model, batch_size=16):
    if not text.strip():
        return "Please input your texts."

    texts = [text]
    translated_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        input_ids = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        output_ids = model.generate(
            **input_ids,
            decoder_start_token_id=tokenizer.lang_code_to_id.get("vi_VN", None),
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        vi = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        translated_texts.extend(vi)

    return translated_texts[0]


iface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=4, label="English Texts"),
    outputs=gr.Textbox(lines=4, label="Vietnamese Translation"),
    title="English to Vietnamese Academic Text Translator",
)

if __name__ == "__main__":
    iface.launch()
