from evaluate import load
rouge = load("rouge")

preds = ["model summary"]
refs = ["gold summary"]

print(rouge.compute(predictions=preds, references=refs))
