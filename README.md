# Category-Aware Controllable Summarisation for Indian News

This repository contains the implementation and research work for
Category-Aware Controllable Summarisation applied to Indian news articles.

## Overview
This work proposes a controllable abstractive summarisation framework
that conditions summary generation on news categories such as Politics,
Sports, and Business using a transformer-based model.

## Dataset
NewsSumm dataset (Indian English news):
https://zenodo.org/record/17670865

## Methodology
- Transformer-based BART encoder–decoder model
- Category control via prefix tokens
- Supervised fine-tuning on NewsSumm dataset
- Evaluation using ROUGE metrics
- Real-time deployment using Gradio

## Repository Structure
- `paper/` : Research paper (PDF)
- `src/` : Training, evaluation, and inference scripts
- `app/` : Gradio web application
- `data/` : Dataset description only
  
## 🌐 Live Demo (Hugging Face Space)
You can test the model here:
https://huggingface.co/spaces/yash222hhh/category-aware-controllable-news-summarization

## License
MIT License
