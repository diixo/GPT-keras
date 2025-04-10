
# https://www.kaggle.com/code/vimalpillai/finetuning-gpt2-model-tensorflow/notebook

from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast, GPT2Config, TFGPT2LMHeadModel, GenerationConfig, PreTrainedTokenizerFast
import tensorflow as tf
import numpy as np
from pathlib import Path
import re


# ---------- hyperparams ----------
batch_size = 64
seq_length = 32
embedding_dim = 384
dff = 384
num_heads = 12
num_layers = 4
# ---------------------------------

epochs = 50
learning_rate = 3e-4

tokenizer_path  = "tokenizer-gpt"

# ---------------------------------

def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []

# ---------------------------------

model_path = f"tokenizer-gpt/emma-gen2-{embedding_dim}-{batch_size}-{seq_length}-{dff}-{num_heads}.h5"

with open("data/processed-austen-emma.txt", "r", encoding='utf-8') as f:
    text = f.readlines()

content = []
for line in text:
    line = line.strip()
    if len(str_tokenize_words(line)) > 4:
        content.append(line)

print(f"Lines: {len(content)}, Batches per epoch: {len(content) // batch_size}")

##########################################################################################

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=50000, initial_alphabet=ByteLevel.alphabet(), special_tokens=[
    "<pad>", "<s>", "</s>", "<unk>", "<mask>"
    ])

tokenizer.train(["data/austen-emma.txt"], trainer)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>",
    pad_token = "<pad>",
    mask_token = "<mask>"
)

fast_tokenizer.save_pretrained(tokenizer_path)

##########################################################################################

tokenizer_gpt = GPT2TokenizerFast.from_pretrained(tokenizer_path)

encodings = tokenizer_gpt(content, padding=True, truncation=True, max_length=seq_length, return_tensors="np")


train_data = encodings["input_ids"][:, :-1]
labels = encodings["input_ids"][:, 1:]
attention_masks = encodings["attention_mask"][:, :-1]

##########################################################################################

ds_tf = tf.data.Dataset.from_tensor_slices((train_data, attention_masks, labels))
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


optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
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


model.summary()
#print(config)

# Making Prediction and Saving Model ###########################################################

def generate_text(prompt: str, model: TFGPT2LMHeadModel, max_length = seq_length, do_sample = True):

    assert(max_length <= seq_length)

    encodings = tokenizer_gpt([prompt], return_tensors='tf')

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
    print(tokenizer_gpt.convert_ids_to_tokens(result))

    #print(tokenizer_gpt.pad_token_id, tokenizer_gpt.bos_token_id, tokenizer_gpt.eos_token_id)
    # use skip_special_tokens=True, because we use padding as special symbol
    return tokenizer_gpt.decode(result, skip_special_tokens=True)


print(generate_text("Emma knows", model))
