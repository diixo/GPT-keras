
# https://www.kaggle.com/code/vimalpillai/finetuning-gpt2-model-tensorflow/notebook


from transformers import GPT2TokenizerFast, GPT2Config, TFGPT2LMHeadModel
import tensorflow as tf
import numpy as np


# ---------- hyperparams ----------
batch_size = 12
seq_length = 100
embedding_dim = 256
dff = 256
num_heads = 4
num_layers = 4
# ---------------------------------

tokenizer_gpt = GPT2TokenizerFast.from_pretrained("tokenizer-gpt")

config = GPT2Config(
    bos_token_id = tokenizer_gpt.bos_token_id,
    eos_token_id = tokenizer_gpt.eos_token_id,
    pad_token_id = tokenizer_gpt.pad_token_id
)

###########################################################
# Text Data Preprocessing #################################
with open("data/austen-emma.txt", "r", encoding='utf-8') as f:
    content = f.readlines()


content_p = []
for c in content:
    if len(c) > 10:
        content_p.append(c.strip())
content_p = " ".join(content_p) + tokenizer_gpt.eos_token

tokenized_content = tokenizer_gpt.encode(content_p)

print("resulted lines =", len(content))
print("tokenized_content =", len(tokenized_content))

examples = []
for i in range(0, len(tokenized_content)):
    examples.append(tokenized_content[i: i + seq_length])

##################### prepare train-data #####################
train_data = []
labels = []
for example in examples:
    if len(example) == seq_length:
        train_data.append(example[:-1])
        labels.append(example[1:])

train_data = np.array(train_data).astype(np.int32)
labels = np.array(labels).astype(np.int32)


dataset = tf.data.Dataset.from_tensor_slices((train_data, labels))

dataset = dataset.shuffle(5000).batch(batch_size, drop_remainder=True)

print(len(dataset))


# Defining Model optimizer, loss metrics and compiling Model ###################################
model = TFGPT2LMHeadModel(config)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

# Fine Tuning the GPT2 Model ###################################################################

epochs = 2
model.fit(dataset, epochs=epochs)

# Making Prediction and Saving Model ###########################################################

def generate_text(start, model):
    input_token_ids = tokenizer_gpt.encode(start, return_tensors='tf')
    output = model.generate(input_token_ids,
        max_length = 200,
        num_beams = 5,
        temperature = 0.7,
        no_repeat_ngram_size = 2,
        num_return_sequences = 1
        )
    return tokenizer_gpt.decode(output[0])


#model.save_weights("tokenizer-gpt/tf_finetuning.h5")

print(generate_text("the", model))
