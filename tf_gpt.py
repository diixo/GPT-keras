import tensorflow as tf
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Config


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#
vocab = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = np.array(vocab)
vocab_size = len(vocab)

###############################
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = tf.constant(encode(text), dtype=tf.int32)
n = int(0.9 * len(data))
###############################

text_as_int = np.array([char_to_idx[c] for c in text], dtype=np.int32)

# hyperparams
seq_length = 64
batch_size = 32
buffer_size = n
embedding_dim = 256
num_heads = 4
dff = 256
num_layers = 4
dropout_rate = 0.1

epochs = 50


def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = (tf.data.Dataset.from_tensor_slices(text_as_int)
           .batch(seq_length + 1, drop_remainder=True)
           .map(split_input_target)
           .shuffle(buffer_size)
           .batch(batch_size, drop_remainder=True))

#
config = GPT2Config(vocab_size=vocab_size, n_positions=seq_length, 
                    n_embd=embedding_dim, n_layer=num_layers, n_head=num_heads, 
                    n_inner=dff)
model = TFGPT2LMHeadModel(config)

#model.summary()

#
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn)

model.fit(dataset, epochs=epochs)


def generate_text(model, start_string, length=100):
    input_eval = [char_to_idx[c] for c in start_string]
    input_eval = tf.convert_to_tensor([input_eval])
    generated_text = []

    for _ in range(length):
        predictions = model(input_eval).logits
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        #
        input_eval = tf.concat([input_eval, [[predicted_id]]], axis=-1)
        input_eval = input_eval[:, -config.n_positions:]

        generated_text.append(idx_to_char[predicted_id])

    return start_string + ''.join(generated_text)



generated = generate_text(model, "ROMEO: ")
print(generated)
