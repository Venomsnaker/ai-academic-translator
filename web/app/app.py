import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "pigpig1524/ml-translator"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="vi_VN")
model.to(device)

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
    
    return " ".join(translated_sentences)

iface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=4, label="English Texts"),
    outputs=gr.Textbox(lines=4, label="Vietnamese Translation"),
    title="English to Vietnamese Academic Text Translator",
)

if __name__ == "__main__":
    iface.launch()
