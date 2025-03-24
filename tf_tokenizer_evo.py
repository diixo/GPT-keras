import tensorflow as tf
from transformers import AutoTokenizer, TFGPT2LMHeadModel, GPT2Config
from pathlib import Path

# ---------- hyperparams ----------
batch_size = 32
seq_length = 64
embedding_dim = 256
dff = 256
num_heads = 4
num_layers = 4
dropout_rate = 0.1
epochs = 10
# ----------------------------------

# Токенизация и подготовка данных
tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")

def create_attention_mask(input_ids):
    # Attention mask: 1 для всех токенов, отличных от паддинга
    return (input_ids != tokenizer.pad_token_id).astype(tf.int32)

def split_input_target(chunk):
    input_chunk = chunk[:-1]
    target_chunk = chunk[1:]
    attention_mask = create_attention_mask(input_chunk)  # создаем attention_mask
    return input_chunk, target_chunk, attention_mask

# Чтение текста из файла
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenizer(text, add_special_tokens=True, truncation=True, padding=True, return_tensors="np")["input_ids"]

# Создание датасета
dataset = (tf.data.Dataset.from_tensor_slices(tokens)
           .batch(seq_length + 1, drop_remainder=True)
           .map(split_input_target)
           .shuffle(10000)
           .batch(batch_size, drop_remainder=True))

# Создание модели
config = GPT2Config(
    vocab_size=tokenizer.vocab_size, 
    n_positions=seq_length + 1,
    n_embd=embedding_dim, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_inner=dff
)

model = TFGPT2LMHeadModel(config)

optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn)


filepath = f"trained.h5"

if Path(filepath).exists():
    model.load_weights(filepath)
else:
    model.fit(dataset, epochs=epochs)
    model.save_weights(filepath)

##########################################################

#####################################
"""
def create_attention_mask(input_ids):
    return (input_ids != tokenizer.pad_token_id).astype(tf.int32)

def split_input_target(chunk):
    input_chunk = chunk[:-1]
    target_chunk = chunk[1:]
    attention_mask = create_attention_mask(input_chunk)
    return input_chunk, target_chunk, attention_mask

dataset = (tf.data.Dataset.from_tensor_slices(tokens)
           .batch(seq_length + 1, drop_remainder=True)
           .map(split_input_target)
           .shuffle(10000)
           .batch(batch_size, drop_remainder=True))"
"""


def create_attention_mask(input_ids):
    attention_mask = (input_ids != tokenizer.pad_token_id).astype(tf.int32)
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    attention_mask = tf.where(input_ids == sep_token_id, tf.zeros_like(attention_mask), attention_mask)
    return attention_mask


def split_input_target(chunk):
    input_chunk = chunk[:-1]
    target_chunk = chunk[1:]
    attention_mask = create_attention_mask(input_chunk)
    return input_chunk, target_chunk, attention_mask


dataset = (tf.data.Dataset.from_tensor_slices(tokens)
           .batch(seq_length + 1, drop_remainder=True)
           .map(split_input_target)
           .shuffle(10000)
           .batch(batch_size, drop_remainder=True))

model.compile(optimizer=optimizer, loss=loss_fn)

if Path(filepath).exists():
    model.load_weights(filepath)
else:
    model.fit(dataset, epochs=epochs)
    model.save_weights(filepath)