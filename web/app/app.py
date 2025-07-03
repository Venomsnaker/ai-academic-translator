import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import fitz
import re
import nltk
from nltk.tokenize import sent_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('punkt')

model_id = "pigpig1524/ml-translator"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="vi_VN")
model.to(device)

def clean_pdf_text(text):
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_pdf_text(text.strip())

def translate(text, model=model):
    if not text.strip():
        return "Please input your texts."
    
    sentences = sent_tokenize(text)
    translated_sentences = []
    
    for sentence in sentences:
        input_ids = tokenizer(sentence, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **input_ids,
            decoder_start_token_id=tokenizer.lang_code_to_id.get("vi_VN", None),
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        translated_sentences.append(translated_text)
    return "\n".join(translated_sentences)

def handle_pdf_upload(pdf_file):
    raw_text = extract_text_from_pdf(pdf_file)
    sentences = sent_tokenize(raw_text)
    return "\n".join(sentences)

with gr.Blocks() as iface:
    gr.Markdown("## English to Vietnamese Academic Text Translator")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(lines=10, label="English Texts", placeholder="Type or upload PDF...")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        with gr.Column(scale=1):
            output_text = gr.Textbox(lines=10, label="Vietnamese Translation")
    
    translate_button = gr.Button("Translate")

    pdf_input.change(fn=handle_pdf_upload, inputs=pdf_input, outputs=text_input)
    translate_button.click(fn=translate, inputs=text_input, outputs=output_text)

if __name__ == "__main__":
    iface.launch()
