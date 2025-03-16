import tensorflow as tf
from tensorflow import keras
from keras import layers
import math

# ---------- data-parameters ----------
batch_size = 32 # amount independent sequences will we process in parallel
block_size = 64 # maximum context length for predictions
max_iters = 5000
#eval_interval = 1000
learning_rate = 5e-4
eval_iters = 100

n_embd = 256
n_head = 4
n_layer = 4

# ---------- static-parameters ----------
dropout_rate = 0.2
random_seed = 2081


tf.random.set_seed(random_seed)

# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
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
def fetch_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = tf.random.uniform(shape=(batch_size,), maxval=len(data_split) - block_size, dtype=tf.int32)
    x = tf.stack([data_split[i: i + block_size] for i in ix])
    y = tf.stack([data_split[i+1: i + block_size+1] for i in ix])
    return x, y


def estimate_loss(model):
    out = {}
    model.trainable = False
    for split in ['train', 'val']:
        losses = tf.zeros(eval_iters, dtype=tf.float32)
        for k in range(eval_iters):
            X, Y = fetch_batch(split)
            logits, loss = model(X, Y)
            losses = tf.tensor_scatter_nd_add(losses, [[k]], [loss])
        out[split] = tf.reduce_mean(losses)
    model.trainable = True
    return out


class CausalSelfAttention(layers.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = layers.Dense(n_embd, 3 * n_embd, bias=False) # (n_embd, 3 * n_embd)
        # output projection
        self.c_proj = layers.Dense(n_embd, n_embd, bias=False)     # (n_embd, n_embd)
        # regularization
        self.attn_dropout = layers.Dropout(dropout_rate)
        self.resid_dropout = layers.Dropout(dropout_rate)
        self.n_head = n_head
        self.n_embd = n_embd


    def call(self, x):
        B, T, C = x.shape   # B = batch size, T = tokens sequence length, C = embedding dimensionality (n_embd)

        # calculate q=query, k=key, v=values for all heads in batch and move head forward to be the batch dim

        q, k, v = tf.split(self.c_attn(x), num_or_size_splits=3, axis=2)
        # c_attn(x) transform (B, T, n_embd) --> (B, T, 3*n_embd)
        # q, k, v <-- (B, T, 3*n_embd).split by (n_embd, dim=2)
        # q, k, v = (B, T, n_embd)

        # transform q,k,v=(B, T, n_embd) -->
        q = tf.reshape(q, (B, T, self.n_head, C // self.n_head))    # (B, T, n_head, head_size)
        q = tf.transpose(q, perm=[0, 2, 1, 3])                      # (B, n_head, T, head_size)

        k = tf.reshape(k, (B, T, self.n_head, C // self.n_head))    # (B, T, n_head, head_size)
        k = tf.transpose(k, perm=[0, 2, 1, 3])                      # (B, n_head, T, head_size)

        v = tf.reshape(v, (B, T, self.n_head, C // self.n_head))    # (B, T, n_head, head_size)
        v = tf.transpose(v, perm=[0, 2, 1, 3])                      # (B, n_head, T, head_size)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        """
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
        """


class Head(layers.Layer):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size

        self.key   = layers.Dense(units=head_size, use_bias=False) # (n_embd, head_size)
        self.query = layers.Dense(units=head_size, use_bias=False) # (n_embd, head_size)
        self.value = layers.Dense(units=head_size, use_bias=False) # (n_embd, head_size)

        tril = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)
        self.tril = tf.constant(tril)
        self.dropout = layers.Dropout(dropout_rate)


    def call(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape   # B = batch size, T = tokens sequence length, C = embedding dimensionality (n_embd)
        #assert(block_size == T)    # TODO: text_generation

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
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = layers.Dense(units=n_embd)  # head_size * num_heads, n_embd
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))

        assert out.shape[1] == x.shape[1] and out.shape[2] == n_embd
        return out


class FeedForward(layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

        self.net = tf.keras.Sequential([
            layers.Dense(units=4*n_embd),  # (n_embd, 4*n_embd)
            layers.ReLU(),
            layers.Dropout(dropout_rate),
            layers.Dense(units=n_embd),    # (4*n_embd, n_embd)
        ])

    def call(self, x):
        out = self.net(x)   # B, T, n_embd
        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd
        return out


class Block(layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        assert n_embd % n_head == 0
        super().__init__()
        head_size = n_embd // n_head

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffwd = FeedForward(n_embd)

        self.dropout_sa = layers.Dropout(dropout_rate)
        self.dropout_ffn = layers.Dropout(dropout_rate)


    def call(self, x, training=False):
        # Pre-LN: normalization before MHA
        x = x + self.dropout_sa(self.sa(self.ln1(x)), training=training)     # dropout output only MHA
        x = x + self.dropout_ffn(self.ffwd(self.ln2(x)), training=training)  # dropout output only FFN
        return x


class BigramLanguageModel(keras.Model):

    def __init__(self, vocab_size):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table    = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = keras.Sequential(
            [Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = layers.LayerNormalization(epsilon=1e-6) # final layer norm
        
        bias_initializer = tf.keras.initializers.RandomNormal(
            mean=0.0,
            stddev=0.02 / math.sqrt(2 * n_layer))
        self.lm_head = layers.Dense(input_shape=(n_embd,), units=vocab_size, bias_initializer=bias_initializer)


    def call(self, idx, targets=None):
        B, T = idx.shape

        if targets is not None: assert(block_size == T)

        # idx and targets are both (B=batch, T=time) tensor of integers
        tok_emb = self.token_embedding_table(idx)               # (B, T, C=n_embd)
        pos_emb = self.position_embedding_table(tf.range(T))    # (T, C=n_embd)
        x = tok_emb + pos_emb               # (B, T, C)
        x = self.blocks(x)                  # (B, T, C)
        x = self.ln_f(x)                    # (B, T, C)
        logits = self.lm_head(x)            # (B, T, vocab_sz)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        return logits, loss


    def generate_text(self, max_new_tokens: int):
        idx = tf.zeros((1, 1), dtype=tf.int64)

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
        return idx.numpy()[0]


def train_model(model: BigramLanguageModel):

    optimizer = tf.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=5e-3,
        epsilon=1e-8)

    for iter in range(max_iters):
        loss = 0
        with tf.GradientTape() as tape:
            # forward pass
            logits, loss = model(*fetch_batch("train"))

        # backward pass
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if iter % 100 == 0:
            print(f"...on {iter}(th) train_loss={loss:.6f}")

    model.summary()
    # Final estimation:
    losses = estimate_loss(model)
    print(f"Finished, steps={iter+1}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")


model = BigramLanguageModel(vocab_size)
train_model(model)

# Generate text from the model
generated_text = decode(model.generate_text(max_new_tokens=500))
print(generated_text)

