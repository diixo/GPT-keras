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

tokenized_text = tokenizer.encode(text, add_special_tokens=True)  # Добавляем специальные токены

# Преобразуем в массив numpy
text_as_int = np.array(tokenized_text, dtype=np.int32)

# Размер словаря из MiniLM
vocab_size = tokenizer.vocab_size

###############################
# stoi = {ch: i for i, ch in enumerate(vocab)}
# itos = np.array(vocab)
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])
# data = tf.constant(encode(text), dtype=tf.int32)
n = int(0.9 * len(text_as_int))
###############################

buffer_size = n

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = (tf.data.Dataset.from_tensor_slices(text_as_int)
           .batch(seq_length + 1, drop_remainder=True)
           .map(split_input_target)
           .shuffle(buffer_size)
           .batch(batch_size, drop_remainder=True))

###############################
config = GPT2Config(vocab_size=vocab_size, n_positions=seq_length, 
                    n_embd=embedding_dim, n_layer=num_layers, n_head=num_heads, 
                    n_inner=dff)
model = TFGPT2LMHeadModel(config)

###############################
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn)

model.fit(dataset, epochs=epochs)

model.summary()


def generate_text(model, start_string, length=50):
    input_eval = tokenizer.encode(start_string, return_tensors="tf")
    
    for _ in range(length):
        predictions = model(input_eval).logits
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        input_eval = tf.concat([input_eval, [[predicted_id]]], axis=-1)

    return tokenizer.decode(input_eval.numpy()[0])


generated = generate_text(model, "ROMEO: ", 100)
print(generated)
