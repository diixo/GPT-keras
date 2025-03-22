import tensorflow as tf
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Config
from pathlib import Path
from transformers import AutoTokenizer

import os


# Загружаем токенайзер MiniLM
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

filepath = f"shakespeare-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# tokenized_text = tokenizer.encode(text, max_length=seq_length, truncation=True, padding="max_length", add_special_tokens=True)

# tokens = tokenizer(text, return_tensors="tf")["input_ids"][0].numpy()

tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

# Разбиваем на куски по seq_length (64)
#chunks = [tokens[i:i+seq_length] for i in range(0, len(tokens), seq_length)]

# Преобразуем в тензоры
#chunks = [tf.convert_to_tensor([chunk]) for chunk in chunks]

##############################

# Преобразуем в массив numpy
#text_as_int = np.array(tokenized_text, dtype=np.int32)

# Размер словаря из MiniLM
vocab_size = tokenizer.vocab_size

###############################
# stoi = {ch: i for i, ch in enumerate(vocab)}
# itos = np.array(vocab)
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])
# data = tf.constant(encode(text), dtype=tf.int32)
#n = int(0.9 * len(text_as_int))
###############################

def create_sequences(tokenized_text, seq_length):
    X, Y = [], []
    for i in range(len(tokenized_text) - seq_length):
        X.append(tokenized_text[i: i + seq_length])
        Y.append(tokenized_text[i + 1: i + seq_length + 1])
    return np.array(X), np.array(Y)

# 🔹 Создаём dataset
X, Y = create_sequences(tokens, seq_length)

dataset = (tf.data.Dataset.from_tensor_slices((X, Y))
           .shuffle(10000)
           .batch(batch_size, drop_remainder=True))

# dataset = (tf.data.Dataset.from_tensor_slices(text_as_int)
#            .batch(seq_length, drop_remainder=True)
#            .shuffle(10000)
#            .batch(batch_size, drop_remainder=True))

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


# def generate_text(model, start_string, length=50):
#     input_eval = tokenizer.encode(start_string, return_tensors="tf")
    
#     for _ in range(length):
#         predictions = model(input_eval).logits
#         predictions = predictions[:, -1, :]
#         predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
#         input_eval = tf.concat([input_eval, [[predicted_id]]], axis=-1)

#     return tokenizer.decode(input_eval.numpy()[0])


def generate_text(model, start_string, length=50):
    input_ids = tokenizer(start_string, return_tensors="tf")["input_ids"]

    for _ in range(length):
        predictions = model(input_ids).logits
        predicted_id = tf.random.categorical(predictions[:, -1, :], num_samples=1).numpy()[0, 0]
        input_ids = tf.concat([input_ids, [[predicted_id]]], axis=-1)

    return tokenizer.decode(input_ids.numpy()[0])


generated = generate_text(model, "ROMEO: ", 100)
print(generated)
