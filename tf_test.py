from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, AutoTokenizer, GPT2TokenizerFast
import tensorflow as tf
from pathlib import Path
import re


# ---------- hyperparams ----------
batch_size = 64
seq_length = 32
embedding_dim = 256
dff = 256
num_heads = 8
num_layers = 4
# ---------------------------------

learning_rate = 5e-4
epochs = 5

# ---------------------------------

def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []


model_path = f"emma-gpt2-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

with open("tokenizer-gpt/austen-emma.txt", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()

lines = [line for line in lines if len(str_tokenize_words(line)) > 1]

batches_per_epoch = len(lines) // batch_size
print(f"Lines: {len(lines)}, Batches per epoch: {batches_per_epoch}")

#####################################################################

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

print(f"model.config: vocab.sz={tokenizer.vocab_size},",
    f"pad_token_id={tokenizer.pad_token_id},",
    f"bos_token_id={tokenizer.bos_token_id},",
    f"eos_token_id={tokenizer.eos_token_id};",
    )

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

##########################################################################################

tokens = tokenizer(lines, padding=True, truncation=True, return_tensors='np', max_length=seq_length + 1)

input_ids = tokens["input_ids"]
attention_masks = tokens["attention_mask"]

##########################################################################################

model = TFGPT2LMHeadModel(config)

optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)


def compute_loss(labels, logits):
    logits = logits[:, :-1, :]
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    return loss


def map_fn(input_chunk, attention_mask):
    input_chunk = input_chunk[:, :-1]
    target_chunk = input_chunk[:, 1:]
    attention_mask = attention_mask[:, :-1]
    return input_chunk, target_chunk, attention_mask


ds_tf = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
dataset = ds_tf.shuffle(5000).batch(batch_size, drop_remainder=True)

model.compile(optimizer=optimizer, loss=compute_loss)

###################################################

if Path(model_path).exists():
    dummy_input = tf.ones((1, seq_length), dtype=tf.int32)
    model(dummy_input)
    model.load_weights(model_path)
else:
    model.fit(dataset.map(map_fn), epochs=epochs)
    model.save_weights(model_path)

# --------------------------------------------------
model.summary()


def generate_text(model, tokenizer: GPT2Tokenizer, prompt: str):

    encoding = tokenizer(prompt, return_tensors='tf')

    if "input_ids" not in encoding or encoding["input_ids"] is None:
        raise ValueError("Ошибка: 'input_ids' не был сгенерирован!")

    _ids = input_ids

    print(f"input_ids type: {type(_ids)}")
    print(f"input_ids shape: {_ids.shape}")
    print(f"input_ids tensor: {_ids}")

    _ids = tf.squeeze(_ids, axis=0)

    decoded_text = tokenizer.decode(_ids.numpy(), skip_special_tokens=True)

    print(f"Decoded text: {decoded_text}")
    
    return decoded_text


#result = generate_text(model, tokenizer, "ROMEO:")

#print(f"Final result: {result}")
