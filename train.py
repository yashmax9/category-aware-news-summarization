from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("your_dataset_name")

def preprocess(batch):
    inputs = [f"<{c}> {t}" for c, t in zip(batch["category"], batch["article"])]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length")
    labels = tokenizer(batch["summary"], truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, batched=True)

args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    report_to="none"
)

trainer = Trainer(model=model, args=args, train_dataset=dataset["train"])
trainer.train()

model.save_pretrained("category-aware-model")
tokenizer.save_pretrained("category-aware-model")
