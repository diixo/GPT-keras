from transformers import TFGPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
import tensorflow as tf
from pathlib import Path
import re


def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []

# ---------- hyperparams --------------------------------------------
batch_size = 32
seq_length = 24
embedding_dim = 384
dff = 384
num_heads = 12
num_layers = 4
# -------------------------------------------------------------------

learning_rate = 5e-4
epochs = 10

#####################################################################

model_path = f"tokenizer-gpt/romeo-gpt2-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

with open("input.txt", "r", encoding="utf-8") as file:
    lines = file.read().split("\n")

lines = [line for line in lines if len(str_tokenize_words(line)) > 2]

batches_per_epoch = len(lines) // batch_size
print(f"Lines: {len(lines)}, Batches per epoch: {batches_per_epoch}")

#####################################################################

tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer-gpt")

print(f"model.config: vocab.sz={tokenizer.vocab_size},",
    f"pad_token_id={tokenizer.pad_token_id},",
    f"bos_token_id={tokenizer.bos_token_id},",
    f"eos_token_id={tokenizer.eos_token_id};",
    )

tokens = tokenizer(lines, add_special_tokens=True, padding=True, truncation=True, max_length=seq_length, return_tensors="np")

input_ids = tokens["input_ids"]
attention_masks = tokens["attention_mask"]


#tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(
    vocab_size=tokenizer.vocab_size, 
    n_positions=seq_length + 1,
    n_embd=embedding_dim, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_inner=dff,

    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)


model = TFGPT2LMHeadModel(config)

optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def compute_loss(labels, logits):
    logits = logits[:, :-1, :]
    return loss_fn(labels, logits)


def map_fn(input_chunk, attention_mask):
    input_chunk = input_chunk[:, :-1]
    target_chunk = input_chunk[:, 1:]
    attention_mask = attention_mask[:, :-1]
    return input_chunk, target_chunk, attention_mask


ds_tf = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
dataset = ds_tf.shuffle(5000).batch(batch_size, drop_remainder=True)

model.compile(optimizer=optimizer, loss=compute_loss)

if Path(model_path).exists():
    dummy_input = tf.ones((1, seq_length), dtype=tf.int32)
    model(dummy_input)
    model.load_weights(model_path)
else:
    model.fit(dataset.map(map_fn), epochs=epochs)
    model.save_weights(model_path)

# -------------------------------------------------------------------
model.summary()


def top_k_filtering(logits, top_k=50, filter_value=-float('Inf')):
    if top_k > 0:
        values, _ = tf.math.top_k(logits, k=top_k)
        min_values = values[:, -1, tf.newaxis]
        logits = tf.where(logits < min_values, filter_value, logits)
    return logits


def generate_text(model: TFGPT2LMHeadModel, tokenizer: GPT2TokenizerFast, prompt: str, length=seq_length, top_k=20):

    input_ids = tokenizer(prompt, return_tensors="tf")["input_ids"]
    print("-->>", input_ids[0].numpy(), tokenizer.convert_ids_to_tokens(input_ids[0]))

    sz = length - input_ids.shape[-1]

    for _ in range(sz):
        predictions = model(input_ids).logits
        predictions = predictions[:, -1, :]

        # -->> extinguish pad token
        pad_token_id = tokenizer.pad_token_id
        logits = tf.tensor_scatter_nd_update(
            predictions,
            indices=[[0, pad_token_id]],  # batch=0
            updates=[-1e8]
        )
        # <<--

        filtered_logits = top_k_filtering(logits, top_k=top_k)
        categoricals = tf.random.categorical(filtered_logits, num_samples=1).numpy()
        predicted_id = categoricals[-1, 0]

        input_ids = tf.concat([input_ids, [[predicted_id]]], axis=-1)

        input_ids = input_ids[:, -config.n_positions:]

    return tokenizer.decode(input_ids[0].numpy(), skip_special_tokens=True)


generated = generate_text(model, tokenizer, "ROMEO: ")
print(generated)
print("*" * 80)
