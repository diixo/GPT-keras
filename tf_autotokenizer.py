from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")

text = ["This is a sentence viiX.", "viiX is about of IT", "Here is another sentence, that is a bit longer."]

# add_special_tokens=True (by default): [CLS]=101, [SEP]=102...
tokens = tokenizer(text, add_special_tokens=False, truncation=False, padding=True, return_tensors="np")["input_ids"] 

print(tokens)
print("max_id:", tokens.max())
print("max_sz:", tokens.shape[-1])

decoded_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in tokens]
print(decoded_texts)
