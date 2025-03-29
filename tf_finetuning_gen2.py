
# https://www.kaggle.com/code/vimalpillai/finetuning-gpt2-model-tensorflow/notebook

from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast, GPT2Config, TFGPT2LMHeadModel, GenerationConfig
import tensorflow as tf
import numpy as np
from pathlib import Path
import re


# ---------- hyperparams ----------
batch_size = 32
seq_length = 32
embedding_dim = 96
dff = 96
num_heads = 4
num_layers = 3
# ---------------------------------

epochs = 5
learning_rate = 3e-4

model_path      = "tokenizer-gpt/tf-finetuning-gen2.h5"
tokenizer_path  = "tokenizer-gpt/tokenizer.json"

# ---------------------------------

def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []


def clean_mask_tokens(encodings, mask_tokens, pad_token_id):

    train_data = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]

    batch_size, seq_len = train_data.shape
    new_train_data = np.full((batch_size, seq_len), pad_token_id, dtype=np.int32)
    new_attention_mask = np.zeros((batch_size, seq_len), dtype=np.int32)

    for i in range(batch_size):
        j = 0
        for k in range(seq_len):
            if train_data[i, k] in mask_tokens:
                continue
            if j < seq_len:
                new_train_data[i, j] = train_data[i, k]
                new_attention_mask[i, j] = attention_masks[i, k]
                j += 1
    encodings["input_ids"] = new_train_data
    encodings["attention_mask"] = new_attention_mask

# ---------------------------------
content = []
with open("tokenizer-gpt/processed-austen-emma.txt", "r", encoding='utf-8') as f:
    text = f.readlines()
content.extend([line.strip() for line in text if len(str_tokenize_words(line)) > 5])


# with open("tokenizer-gpt/austen.txt", "r", encoding='utf-8') as f:
#     text = f.readlines()
# content.extend([line.strip() for line in text if len(str_tokenize_words(line)) > 5])

print(f"size={len(content)}")
# ---------------------------------

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=50000, initial_alphabet=ByteLevel.alphabet(), special_tokens=[
    "<pad>", "<s>", "</s>", "<unk>", "<mask>"
    ])

tokenizer.train(["tokenizer-gpt/austen-emma.txt"], trainer)

tokenizer.save(tokenizer_path)

tokenizer_gpt = GPT2TokenizerFast.from_pretrained("tokenizer-gpt")

tokenizer_gpt.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>"
})

# Text Data Preprocessing #################################

mask_tokens_ids = tokenizer_gpt.convert_tokens_to_ids(['a', 'Ġa', 'an', 'Ġan', 'the', 'Ġthe'])

mask_tokens = tokenizer_gpt.convert_ids_to_tokens(mask_tokens_ids)

##########################################################################################

encodings = tokenizer_gpt(content, padding="max_length", truncation=True, max_length=seq_length, return_tensors="np")

clean_mask_tokens(encodings, set(mask_tokens_ids), tokenizer_gpt.pad_token_id)


train_data = encodings["input_ids"][:, :-1]
labels = encodings["input_ids"][:, 1:]
attention_masks = encodings["attention_mask"][:, :-1]

##########################################################################################

ds_tf = tf.data.Dataset.from_tensor_slices((train_data, labels, attention_masks))
dataset = ds_tf.shuffle(5000).batch(batch_size, drop_remainder=True)

def train_step(x, mask, y):
    return {"input_ids": x, "attention_mask": mask}, y

# Defining Model optimizer, loss metrics and compiling Model ###################################

config = GPT2Config(
    n_positions=seq_length,
    n_embd=embedding_dim,
    n_layer=num_layers,
    n_head=num_heads,
    n_inner=dff,

    vocab_size=tokenizer_gpt.vocab_size,
    bos_token_id=tokenizer_gpt.bos_token_id,
    eos_token_id=tokenizer_gpt.eos_token_id,
    pad_token_id=tokenizer_gpt.pad_token_id
)

model = TFGPT2LMHeadModel(config)

print(f"model.config: vocab.sz={tokenizer_gpt.vocab_size},",
    f"pad_token_id={model.config.pad_token_id},",
    f"bos_token_id={model.config.bos_token_id},",
    f"eos_token_id={model.config.eos_token_id}",
    )


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

###################################################

if Path(model_path).exists():
    dummy_input = tf.ones((1, seq_length), dtype=tf.int32)
    model(dummy_input)
    model.load_weights(model_path)
else:
    model.fit(dataset.map(train_step), epochs=epochs)
    model.save_weights(model_path)
    #model.save("my_gpt2")

model.summary()

# Making Prediction and Saving Model ###########################################################

def generate_text(prompt: str, model: TFGPT2LMHeadModel, max_length = seq_length):

    assert(max_length <= seq_length)

    encodings = tokenizer_gpt([prompt], return_tensors='tf')

    gen_config = GenerationConfig(
        max_length = max_length,
        do_sample = True,
        temperature = 0.9,
        top_k = 20,
        top_p = 0.9,
        repetition_penalty = 1.2,
        no_repeat_ngram_size = 1
    )

    output = model.generate(
        inputs = encodings['input_ids'],
        attention_mask = encodings['attention_mask'],
        generation_config = gen_config
    )
    #print(tokenizer_gpt.pad_token_id, tokenizer_gpt.bos_token_id, tokenizer_gpt.eos_token_id)
    # use add_special_tokens=True, because we use padding as special symbol
    return tokenizer_gpt.decode(output[0], skip_special_tokens=True)


print(generate_text("Emma knew", model))
