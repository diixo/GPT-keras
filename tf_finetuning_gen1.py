from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, GPT2TokenizerFast, GenerationConfig
import tensorflow as tf
from pathlib import Path
import re
import numpy as np


# ---------- hyperparams ----------
batch_size = 64
seq_length = 24
embedding_dim = 384
dff = 384
num_heads = 12
num_layers = 4
# ---------------------------------

learning_rate = 5e-4
epochs = 10

# ---------------------------------

model_path = f"tokenizer-gpt/emma-gen1-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

with open("data/austen-emma.txt", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()

lines = [line.lower() for line in lines if len(line.split()) > 2]

batches_per_epoch = len(lines) // batch_size
print(f"Lines: {len(lines)}. Batches per epoch: {batches_per_epoch}")

##########################################################################################

tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer-gpt")

print(f"model.config: vocab.sz={tokenizer.vocab_size},",
    f"pad_token_id={tokenizer.pad_token_id},",
    f"bos_token_id={tokenizer.bos_token_id},",
    f"eos_token_id={tokenizer.eos_token_id};",
    )

##########################################################################################

tokens = tokenizer(lines, padding=True, truncation=True, return_tensors='np', max_length=seq_length + 1)

input_ids = tokens["input_ids"]
attention_masks = tokens["attention_mask"]

assert(np.max(input_ids) < tokenizer.vocab_size)

##########################################################################################

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

class ShiftedSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[:, :-1, :]  # [batch, seq-1, vocab]
        return super().update_state(y_true, y_pred, sample_weight)

metric = ShiftedSparseCategoricalAccuracy("accuracy")
model.compile(optimizer=optimizer, loss=compute_loss, metrics=[metric])

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


def generate_text(prompt: str, model: TFGPT2LMHeadModel, tokenizer: GPT2TokenizerFast, max_length=seq_length, do_sample=True):

    encodings = tokenizer([prompt], return_tensors='tf')

    if "input_ids" not in encodings or encodings["input_ids"] is None:
        raise ValueError("Error: 'input_ids' have not been generated!")

    if do_sample: # rnd
        gen_config = GenerationConfig(
            max_length = seq_length,
            do_sample = do_sample,
            temperature = 0.8,
            top_k = 20,
            top_p = 0.9,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 1
        )
    else:
        gen_config = GenerationConfig(
            max_length = seq_length,
            do_sample = do_sample,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 1
        )

    output = model.generate(
        inputs = encodings['input_ids'],
        attention_mask = encodings['attention_mask'],
        generation_config = gen_config
    )
    generated = tf.expand_dims(output[0], axis=0)

    while generated.shape[1] < max_length:
        input_slice = generated[:, -seq_length + 1:]

        # -->>
        #slice_ids = tf.reshape(input_slice, [-1])
        #print(slice_ids.numpy())
        #print(tokenizer.convert_ids_to_tokens(slice_ids))
        # <<--

        logits = model(input_slice).logits[:, -1, :]

        # masking paddings
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            logits = tf.tensor_scatter_nd_update(
                logits,
                indices=[[0, pad_id]],
                updates=[-1e10]
            )

        next_token_id = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
        #-->>
        #print(next_token.shape, "=", tokenizer.convert_ids_to_tokens(next_token))
        #print("value=", tf.squeeze(next_token).numpy())
        #<<--
        generated = tf.concat([generated, next_token_id], axis=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


result = generate_text(prompt="Emma knows ", model=model, tokenizer=tokenizer)

print(f"Final result: {result}")
