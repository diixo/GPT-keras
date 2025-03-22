import tensorflow as tf
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Config
from pathlib import Path


epochs = 20

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

###############################
vocab = sorted(set(text))
vocab_size = len(vocab)

###############################
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = np.array(vocab)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = tf.constant(encode(text), dtype=tf.int32)
n = int(0.9 * len(data))
###############################

buffer_size = n
text_as_int = np.array(encode(text), dtype=np.int32)


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


if Path(filepath).exists():
    dummy_input = tf.ones((1, seq_length), dtype=tf.int32)
    model(dummy_input)
    model.load_weights(filepath)
else:
    model.fit(dataset, epochs=epochs)
    model.save_weights(filepath)

model.summary()


def generate_text(model, start_string, length=100):
    input_eval = tf.convert_to_tensor([encode(start_string)])
    generated_text = []

    for _ in range(length):
        predictions = model(input_eval).logits
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.concat([input_eval, [[predicted_id]]], axis=-1)
        input_eval = input_eval[:, -config.n_positions:]

        generated_text.append(itos[predicted_id])

    return start_string + ''.join(generated_text)


generated = generate_text(model, "ROMEO: ", 500)
print(generated)
