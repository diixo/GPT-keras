
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Config, AutoTokenizer
import re


def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []

# ---------- hyperparams ----------
batch_size = 32
seq_length = 64
embedding_dim = 256
dff = 256
num_heads = 4
num_layers = 4
# ---------------------------------

dropout_rate = 0.1
learning_rate = 5e-4
epochs = 3

#####################################################################

with open("input.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

lines = [line for line in lines if len(str_tokenize_words(line)) > 1]

batches_per_epoch = len(lines) // batch_size
print(f"Lines: {len(lines)}, Batches per epoch: {batches_per_epoch}")

#####################################################################

tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size, 
    n_positions=seq_length,
    n_embd=embedding_dim, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_inner=dff
)


model = TFGPT2LMHeadModel(config)

optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(labels, logits):
    logits = logits[:, :-1, :]
    return loss_fn(labels, logits)


tokens = tokenizer(lines, add_special_tokens=True, padding="max_length", truncation=True, max_length=seq_length, return_tensors="np")

input_ids = tokens["input_ids"]
attention_masks = tokens["attention_mask"]


def map_fn(input_chunk, attention_mask):
    input_chunk = input_chunk[:-1]
    target_chunk = input_chunk[1:]
    attention_mask = attention_mask[:-1]
    return input_chunk, target_chunk, attention_mask


dataset = (tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
           .map(map_fn)
           .shuffle(10000)
           .batch(batch_size, drop_remainder=True))

model.compile(optimizer=optimizer, loss=loss)

model.fit(dataset, epochs=epochs)

# --------------------------------------------------
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
