import gradio as gr
from transformers import pipeline

MODEL_NAME = "yash222hhh/category-aware-news-summarizer"

summarizer = pipeline("summarization", model=MODEL_NAME)

def summarize(text, category):
    prompt = f"<{category}> {text}"
    output = summarizer(prompt, max_length=150, min_length=50, do_sample=False)
    return output[0]["summary_text"]

demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(lines=12, label="News Article"),
        gr.Dropdown(["Politics", "Sports", "Business"], label="Category")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="Category-Aware News Summarization",
    description="Generates category-controlled summaries for Indian news articles."
)

demo.launch()
