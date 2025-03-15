import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


# ---------- hyperparameters ----------
batch_size = 32 # amount independent sequences will we process in parallel
block_size = 80 # maximum context length for predictions

max_iters = 5000
eval_interval = 1000
learning_rate = 1e-3
eval_iters = 200

n_embd = 256
n_head = 4
n_layer = 4

dropout_rate = 0.2
random_seed = 2081

tf.random.set_seed(random_seed)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = tf.constant(encode(text), dtype=tf.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = tf.random.uniform(shape=(batch_size,), maxval=len(data_split) - block_size, dtype=tf.int32)
    x = tf.stack([data_split[i:i+block_size] for i in ix])
    y = tf.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y


class Head(layers.Layer):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size

    def build(self, input_shape):
        assert(block_size == input_shape[1])
        self.key   = layers.Dense(units=self.head_size, use_bias=False) # (n_embd, head_size)
        self.query = layers.Dense(units=self.head_size, use_bias=False) # (n_embd, head_size)
        self.value = layers.Dense(units=self.head_size, use_bias=False) # (n_embd, head_size)

        tril = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)
        self.tril = tf.constant(tril)
        self.dropout = layers.Dropout(dropout_rate)


    def call(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)                         # (B, T, head_size)
        q = self.query(x)                       # (B, T, head_size)
        k_T = tf.transpose(k, perm=[0, 2, 1])   # (B, T, head_size) --> (B, head_size, T)

        # compute attention scores ("affinities")
        scale = tf.math.rsqrt(tf.cast(k.shape[-1], tf.float32))
        wei = tf.matmul(q, k_T) * scale         # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        wei = tf.where(self.tril[:T, :T] == 0, float('-inf'), wei)  # (B, T, T)
        wei = tf.nn.softmax(wei, axis=-1)       # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)                       # (B, T, head_size)
        out = tf.matmul(wei, v)                 # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)

        assert out.shape[1] == x.shape[1] and out.shape[2] == self.head_size
        return out


class MultiHeadAttention(layers.Layer):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

    def build(self, input_shape):
        self.heads = [Head(self.head_size) for _ in range(self.num_heads)]
        self.proj = layers.Dense(units=n_embd)  # (head_size * num_heads, n_embd)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out), training=training)

        assert out.shape[1] == x.shape[1] and out.shape[2] == n_embd
        return out


class FeedForward(layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

    def build(self, input_shape):
        self.net = tf.keras.Sequential([
            layers.Dense(units=4*self.n_embd),  # (n_embd, 4*n_embd)
            layers.ReLU(),
            layers.Dropout(dropout_rate),
            layers.Dense(units=self.n_embd),    # (4*n_embd, n_embd)
        ])

    def call(self, x, training=False):
        out = self.net(x, training=training)   # B, T, n_embd
        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd
        return out


class Block(layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.head_size = n_embd // n_head
        self.n_head = n_head
        self.n_embd = n_embd

    def build(self, input_shape):
        self.sa = MultiHeadAttention(self.n_head, self.head_size)
        self.ffwd = FeedForward(self.n_embd)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_sa = layers.Dropout(dropout_rate)
        self.dropout_ffn = layers.Dropout(dropout_rate)


    def call(self, x, training=False):
        # Pre-LN: normalization before MHA
        x = x + self.dropout_sa(self.sa(self.ln1(x)), training=training)     # dropout output only MHA
        x = x + self.dropout_ffn(self.ffwd(self.ln2(x)), training=training)  # dropout output only FFN
        return x


class BigramLanguageLayer(layers.Layer):

    def __init__(self, vocab_size, n_embd, n_head, n_block):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_block = n_block

    def build(self, input_shape):
        assert(block_size == input_shape[1])
        self.block_size = input_shape[1]

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table    = layers.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = layers.Embedding(self.block_size, self.n_embd)
        self.blocks = keras.Sequential([Block(self.n_embd, self.n_head) for _ in range(self.n_block)])
        self.ln_f = layers.LayerNormalization(epsilon=1e-6) # final layer norm
        self.lm_head = layers.Dense(units=self.vocab_size)  # (n_embd, vocab_size)


    def call(self, idx, targets=None):
        B, T = idx.shape
        assert(self.block_size == T)

        # idx and targets are both (B=batch, T=time) tensor of integers
        tok_emb = self.token_embedding_table(idx)               # (B, T, C=n_embd)
        pos_emb = self.position_embedding_table(tf.range(0, T)) # (T, C=n_embd)
        x = tok_emb + pos_emb               # (B, T, C)
        x = self.blocks(x)                  # (B, T, C)
        x = self.ln_f(x)                    # (B, T, C)
        logits = self.lm_head(x)            # (B, T, vocab_sz)

        if targets is None:
            return logits
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
            return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in tf.range(max_new_tokens):
            # crop idx to the last block_size token
            idx_cond = idx[:, -block_size:]
            # get the prediction by logits, loss ignored. 
            logits, loss = self(idx_cond)   # forward to get logits = (B, T, C)
            # focus only on last time step
            logits = logits[:, -1, :]       # becomes (B, C)
            # apply softmax to get max probability
            probs = tf.nn.softmax(logits, axis=-1)  # (B, C)
            # get next char-index
            idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1, dtype=tf.int64) # (B, 1)
            # concatenate to stream of integer indices
            idx = tf.concat([idx, idx_next], axis=1)    # (B, T+1)
        return idx


class TransformerModel:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, learning_rate):
        self.block_size = block_size
        self.vocab_size = vocab_size

        keras.utils.set_random_seed(random_seed)
        inputs = keras.Input((block_size,), dtype="int32")
        outputs = BigramLanguageLayer(vocab_size, n_embd, n_head, n_layer)(inputs)
        self.model = keras.Model(inputs, outputs)

        # for var in self.model.trainable_variables:
        #     print(f"--- {var.name}: {var.shape}")

        # keras.optimizers.experimental.AdamW behaves strangely. Using Adam instead for now.
        self.model.compile(
            optimizer=tf.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-2,
                epsilon=1e-7),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self.model.summary()


    def estimate_loss(self, num_iters: int) -> dict:
        out = {}
        self.model.trainable = False
        for split in ["train", "val"]:
            losses = tf.zeros(eval_iters, dtype=tf.float32)
            for k in range(eval_iters):
                loss = self.model.evaluate(*get_batch(split), verbose=0)
                losses = tf.tensor_scatter_nd_add(losses, [[k]], [loss])
            out[split] = tf.reduce_mean(losses)
        self.model.trainable = True
        return out


    def train_on_batch(self, x, y, *args, **kwargs):
        return self.model.train_on_batch(x, y, *args, **kwargs)


def train_model():
    """ return TransformerModel """

    model = TransformerModel(vocab_size, n_embd, n_head, n_layer, block_size, learning_rate)

    for iter in tf.range(max_iters):

        Xb, Yb = get_batch("train")

        model.train_on_batch(Xb, Yb)

        if (iter % eval_interval == 0):
            losses = model.estimate_loss(eval_iters)
            print(f"...on {iter.numpy()}(th): train_loss({losses['train']:.4f}), val_loss({losses['val']:.4f})")
        else:
            if (iter % 100 == 0):
                print(f"...on {iter}(th) epoch...")


    # final estimation:
    losses = model.estimate_loss(eval_iters)
    print(f"Final step {iter.numpy()}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")
    return model


# Generate text from the model
idx = tf.zeros((1, 1), dtype=tf.int64)  # (B, T)

model = train_model()
generated_text = decode(model.generate(idx, max_new_tokens=500).numpy()[0])
print(generated_text)
