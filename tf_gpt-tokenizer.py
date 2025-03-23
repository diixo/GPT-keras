import tensorflow as tf
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Config
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")

epochs = 10

# ---------- hyperparams ----------
batch_size = 32
seq_length = 64
embedding_dim = 256
dff = 256
num_heads = 4
num_layers = 4
dropout_rate = 0.1
#----------------------------------

filepath = f"shakespeare-miniLM-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

print(len(tokens))

vocab_size = tokenizer.vocab_size

batches_per_epoch = len(tokens) // (batch_size * (seq_length + 1))
print(f"Batches per epoch: {batches_per_epoch}")

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

if Path(filepath).exists():
    dummy_input = tf.ones((1, seq_length), dtype=tf.int32)
    model(dummy_input)
    model.load_weights(filepath)
else:
    model.fit(dataset, epochs=epochs)
    model.save_weights(filepath)

model.summary()

def generate_text(model, start_string, length=50):
    input_ids = tokenizer(start_string, return_tensors="tf")["input_ids"]
    generated_ids = []

    for _ in range(length):
        predictions = model(input_ids).logits
        predictions = predictions[:, -1, :]
        categoricals = tf.random.categorical(predictions, num_samples=1).numpy()
        predicted_id = categoricals[-1, 0]

        input_ids = tf.concat([input_ids, [[predicted_id]]], axis=-1)
        input_ids = input_ids[:, -config.n_positions:]

        generated_ids.append(predicted_id)

    tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    #print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    # Insert spaces
    text = start_string
    for token in tokens:
        if token.startswith("##"):
            text += token[2:]
        else:
            text += " " + token
    return text.strip()


generated = generate_text(model, "ROMEO: ")
print(generated)
