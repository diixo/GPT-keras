import tensorflow as tf
from transformers import AutoTokenizer


batch_size = 1
tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")

text = ["This is a sentence viiX.", "viiX is about of IT", "Here is another sentence, that is a bit longer."]

# add_special_tokens=True (by default): [CLS]=101, [SEP]=102...
tokens = tokenizer(text, add_special_tokens=False, truncation=False, padding=True, return_tensors="np")

input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

print(input_ids)
print("max_id:", input_ids.max())
print("max_sz:", input_ids.shape[-1])

decoded_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in input_ids]
print(decoded_texts)
