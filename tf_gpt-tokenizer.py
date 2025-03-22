import tensorflow as tf
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Config
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")

epochs = 1

# ---------- hyperparams ----------
batch_size = 32
seq_length = 64
embedding_dim = 256
dff = 256
num_heads = 4
num_layers = 4
dropout_rate = 0.1
#----------------------------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

print(len(tokens))

vocab_size = tokenizer.vocab_size

# def create_sequences(tokenized_text, seq_length):
#     X, Y = [], []
#     for i in range(len(tokenized_text) - seq_length):
#         X.append(tokenized_text[i: i + seq_length])
#         Y.append(tokenized_text[i + 1: i + seq_length + 1])
#     return np.array(X), np.array(Y)


# X, Y = create_sequences(tokens, seq_length)

steps_per_epoch = len(tokens) // (batch_size * (seq_length + 1))  # количество батчей на одну эпоху
print(f"Количество батчей в эпохе: {steps_per_epoch}")

#########################################################

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = (tf.data.Dataset.from_tensor_slices(tokens)
           .batch(seq_length + 1, drop_remainder=True)
           .map(split_input_target)
           .shuffle(10000)
           .batch(batch_size, drop_remainder=True))

###############################
config = GPT2Config(
    vocab_size=vocab_size, 
    n_positions=seq_length + 1,
    n_embd=embedding_dim, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_inner=dff
)
model = TFGPT2LMHeadModel(config)


###############################
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn)

model.fit(dataset, epochs=epochs)

model.summary()

def generate_text(model, start_string, length=50):
    input_ids = tokenizer(start_string, return_tensors="tf")["input_ids"]

    for _ in range(length):
        predictions = model(input_ids).logits
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1).numpy()[0, 0]
        input_ids = tf.concat([input_ids, [[predicted_id]]], axis=-1)

    return tokenizer.decode(input_ids.numpy()[0])


generated = generate_text(model, "ROMEO: ", 100)
print(generated)
