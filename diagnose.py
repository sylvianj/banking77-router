from src.predict import load_model
import torch

tokenizer, model, label_names = load_model()

print("=== Model Config ===")
print(f"Model type:           {type(model).__name__}")
print(f"num_labels:           {model.config.num_labels}")
print(f"classifier shape:     {model.classifier.weight.shape}")
print(f"pre_classifier shape: {model.pre_classifier.weight.shape}")
print(f"Total parameters:     {sum(p.numel() for p in model.parameters()):,}")

print("\n=== Tokenization ===")
text = "I lost my card yesterday"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Token IDs:   {inputs['input_ids'][0][:15].tolist()}")
print(f"Decoded:     {tokenizer.decode(inputs['input_ids'][0])}")

print("\n=== Forward Pass ===")
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)

print(f"Logits min:  {logits.min().item():.4f}")
print(f"Logits max:  {logits.max().item():.4f}")
print(f"Logits mean: {logits.mean().item():.4f}")
print(f"Logits std:  {logits.std().item():.4f}")

print("\n=== Top 5 predictions ===")
top5_p, top5_i = torch.topk(probs, 5)
for p, i in zip(top5_p.tolist(), top5_i.tolist()):
    print(f"  {label_names[i]:45s} {p:.4f}")

print("\n=== Label names sanity ===")
print(f"First 3 label names: {label_names[:3]}")
print(f"Lost_or_stolen_card index in list: {label_names.index('lost_or_stolen_card')}")