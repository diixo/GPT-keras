from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, AutoTokenizer, GPT2TokenizerFast, GenerationConfig
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
epochs = 30
# ---------------------------------

model_path = f"tokenizer-gpt/emma-gpt2-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

with open("data/austen-emma.txt", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()

lines = [line.lower() for line in lines if len(line.split()) > 2]

batches_per_epoch = len(lines) // batch_size
print(f"Lines: {len(lines)}. Batches per epoch: {batches_per_epoch}")

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


def generate_text(model: TFGPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str, max_length = seq_length, do_sample = True):

    assert(max_length <= seq_length)

    encodings = tokenizer([prompt], return_tensors='tf')

    if "input_ids" not in encodings or encodings["input_ids"] is None:
        raise ValueError("Error: 'input_ids' have not been generated!")

    if do_sample:
        gen_config = GenerationConfig(
            max_length = max_length,
            do_sample = do_sample,
            temperature = 0.8,
            top_k = 20,
            top_p = 0.9,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 1
        )
    else:
        gen_config = GenerationConfig(
            max_length = max_length,
            do_sample = do_sample,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 1
        )

    output = model.generate(
        inputs = encodings['input_ids'],
        attention_mask = encodings['attention_mask'],
        generation_config = gen_config
    )
    result = output[0]
    #tokenizer.convert_ids_to_tokens(result, skip_special_tokens=True)

    return tokenizer.decode(result, skip_special_tokens=True)


result = generate_text(model, tokenizer, "Emma knows")

print(f"Final result: {result}")
