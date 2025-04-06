
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


seq_length = 32

tokenizer_path  = "tokenizer-test"

# ---------------------------------
def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []
# ---------------------------------

with open("tokenizer-gpt/austen-emma.txt", "r", encoding='utf-8') as f:
    text = f.readlines()

content = []
for line in text:
    line = line.strip()
    if len(str_tokenize_words(line)) > 4:
        content.append(line)

##########################################################################################

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=50000, initial_alphabet=ByteLevel.alphabet(), special_tokens=[
    "<pad>", "<s>", "</s>", "<unk>", "<mask>"
    ])

tokenizer.train(["tokenizer-gpt/austen-emma.txt"], trainer)

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

print(f"model.config: vocab.sz={tokenizer_gpt.vocab_size},",
    f"pad_token_id={tokenizer_gpt.pad_token_id},",
    f"bos_token_id={tokenizer_gpt.bos_token_id},",
    f"eos_token_id={tokenizer_gpt.eos_token_id}")

##########################################################################################


def test(txt = "Does Mrs. Churchill do the same?"):
    ids = tokenizer_gpt(txt, padding=True, truncation=True, max_length=seq_length, return_tensors="np")["input_ids"]
    ids = ids[0]
    print(tokenizer_gpt.convert_ids_to_tokens(ids))

test("Two mixing freaks")

# for line in content:
#     ids = tokenizer_gpt(line, padding=True, truncation=True, max_length=seq_length, return_tensors="np")["input_ids"]
#     ids = ids[0]
#     if 10011 in set(ids):
#         print(line)
#         print(tokenizer_gpt.convert_ids_to_tokens(ids))

