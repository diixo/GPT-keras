from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf
from pathlib import Path
import re


def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []


batch_size = 32
seq_length = 32
embedding_dim = 256
dff = 256
num_heads = 4
num_layers = 4

epochs = 3


model_path = f"shakespeare-gpt2-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

file_path = 'input.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    texts = file.read().splitlines()

lines = [line for line in texts if len(str_tokenize_words(line)) > 1]


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


encodings = tokenizer(lines, padding=True, truncation=True, return_tensors='tf', max_length=seq_length)


config = GPT2Config(
    vocab_size=tokenizer.vocab_size, 
    n_positions=seq_length,
    n_embd=embedding_dim, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_inner=dff
)
model = TFGPT2LMHeadModel(config)

input_ids = encodings["input_ids"]
attention_masks = encodings["attention_mask"]


def compute_loss(labels, logits):
    logits = logits[:, :-1, :]
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    return loss


def split_input_target(input_chunk, attention_mask):
    input_chunk = input_chunk[:-1]
    target_chunk = input_chunk[1:]
    attention_mask = attention_mask[:-1]
    return input_chunk, target_chunk, attention_mask


def map_fn(input_chunk, attention_mask):
    return split_input_target(input_chunk, attention_mask)


dataset = (tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
           .map(map_fn)
           .shuffle(10000)
           .batch(batch_size, drop_remainder=True))


optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4)


model.compile(optimizer=optimizer, loss=compute_loss)

if Path(model_path).exists():
    dummy_input = tf.ones((1, seq_length), dtype=tf.int32)
    model(dummy_input)
    model.load_weights(model_path)
else:
    model.fit(dataset, epochs=epochs)
    model.save_weights(model_path)

model.summary()

