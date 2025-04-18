import tensorflow as tf
from tensorflow import keras
from keras import layers
import math

# ---------- data-parameters ----------
batch_size = 32 # amount independent sequences will we process in parallel
block_size = 64 # maximum context length for predictions
max_iters = 5000
learning_rate = 3e-4
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


class GELU(layers.Layer):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def call(self, x):
        return 0.5 * x * (1 + tf.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))


class CausalSelfAttention(layers.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = layers.Dense(units=3*n_embd, use_bias=False)  # (n_embd, 3*n_embd)
        # output projection
        self.c_proj = layers.Dense(units=n_embd, use_bias=False)    # (n_embd, n_embd)
        # regularization
        self.attn_dropout = layers.Dropout(dropout_rate)
        self.resid_dropout = layers.Dropout(dropout_rate)
        self.n_head = n_head
        self.n_embd = n_embd

        tril = tf.linalg.band_part(tf.ones((1, 1, block_size, block_size)), -1, 0)
        self.tril = tf.constant(tril)


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

        ########################################
        # causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        """
        q: (B, nh, T, hs)
        k: (B, nh, T, hs)
        v: (B, nh, T, hs)
        tril: (1, 1, T, T)
        """

        # (B, nh, T, hs) --> (B, nh, hs, T)
        k_T = tf.transpose(k, perm=[0, 1, 3, 2])

        d_k = tf.cast(k.shape[-1], dtype=tf.float32)
        scale = tf.math.rsqrt(d_k)

        att = tf.matmul(q, k_T) * scale  # (B, nh, T, T)
        #att = tf.where(self.tril[:T, :T] == 0, float('-inf'), att)
        #att = tf.where(self.tril == 0, float('-inf'), att)
        att = tf.where(self.tril[:, :, :T, :T] == 0, float('-inf'), att)
        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = tf.matmul(att, v)   # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)

        # (B, nh, T, hs) --> (B, T, nh, hs)
        y = tf.transpose(y, perm=[0, 2, 1, 3])

        # concat all heads --> (B, T, C), for C = nh * hs
        y = tf.reshape(y, [y.shape[0], y.shape[1], -1])
        y = self.resid_dropout(self.c_proj(y))

        assert(B == y.shape[0] and T == y.shape[1] and C == y.shape[2])
        return y


class FeedForward(layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

        self.net = tf.keras.Sequential([
            layers.Dense(units=4*n_embd),  # (n_embd, 4*n_embd)
            GELU(),
            layers.Dense(units=n_embd),    # (4*n_embd, n_embd)
            layers.Dropout(dropout_rate),
        ])

    def call(self, x):
        out = self.net(x)   # B, T, n_embd
        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd
        return out


class FeedForwardConv(layers.Layer):
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.net = keras.Sequential([
            layers.Conv1D(filters=4, kernel_size=1, activation=self.gelu),
            layers.Conv1D(filters=self.n_embd, kernel_size=1),
            layers.Dropout(dropout_rate)
        ])

    def gelu(self, x):
        return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))

    def call(self, x, *args, **kwargs):
        out = self.net(x)   # B, T, n_embd
        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd
        return out


class FeedForwardHybrid(layers.Layer):

    def __init__(self, n_embd, dropout_rate=0.1):
        super().__init__()
        self.n_embd = n_embd

        self.dense_net = keras.Sequential([
            layers.Dense(units=4 * n_embd), # (n_embd, 4*n_embd)
            GELU(),
            layers.Dense(units=n_embd),     # (4*n_embd, n_embd)
            layers.Dropout(dropout_rate),
        ])

        self.conv_net = keras.Sequential([
            layers.Conv1D(filters=4 * n_embd, kernel_size=1),
            GELU(),
            layers.Conv1D(filters=n_embd, kernel_size=1),
            layers.Dropout(dropout_rate)
        ])

        self.norm = layers.LayerNormalization()
        self.final_dense = layers.Dense(units=n_embd)


    def call(self, x):
        dense_out = self.dense_net(x)       # global neurons connections
        conv_out = self.conv_net(x)         # local neurons connections
        combined = dense_out + conv_out
        normalized = self.norm(combined)
        return self.final_dense(normalized)


class FeedForwardHybridConvDense(layers.Layer):

    def __init__(self, n_embd, dropout_rate=0.1):
        super().__init__()
        self.n_embd = n_embd

        self.conv_net = keras.Sequential([
            layers.Conv1D(filters=4 * n_embd, kernel_size=1),
            GELU(),
            layers.Conv1D(filters=n_embd, kernel_size=1),
            layers.Dropout(dropout_rate),
        ])

        self.dense_net = keras.Sequential([
            layers.Dense(units=4 * n_embd),  # (n_embd, 4*n_embd)
            GELU(),
            layers.Dense(units=n_embd),      # (4*n_embd, n_embd)
            layers.Dropout(dropout_rate),
        ])

        #self.final_dense = layers.Dense(units=n_embd)

    def call(self, x):
        conv_out = self.conv_net(x)

        dense_out = self.dense_net(conv_out)
        return dense_out

        #combined = conv_out + dense_out
        #return self.final_dense(combined)


class Block(layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        assert n_embd % n_head == 0
        super().__init__()

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.csa = CausalSelfAttention(n_embd, n_head)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffwd = FeedForward(n_embd)
        #self.ffwd = FeedForwardHybridConvDense(n_embd)

        self.dropout_sa = layers.Dropout(dropout_rate)
        self.dropout_ffn = layers.Dropout(dropout_rate)


    def call(self, x):
        # Pre-LN: normalization before MHA
        x = x + self.dropout_sa(self.csa(self.ln1(x)))      # dropout output only MHA
        x = x + self.dropout_ffn(self.ffwd(self.ln2(x)))    # dropout output only FFN
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
        self.drop = layers.Dropout(dropout_rate)


    def call(self, idx, targets=None):
        B, T = idx.shape

        if targets is not None: assert(block_size == T)

        # idx and targets are both (B=batch, T=time) tensor of integers
        tok_emb = self.token_embedding_table(idx)               # (B, T, C=n_embd)
        pos_emb = self.position_embedding_table(tf.range(T))    # (T, C=n_embd)
        x = tok_emb + pos_emb               # (B, T, C)
        x = self.drop(x)                    # (B, T, C)
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


    def generate_text(self, prompt: str, max_new_tokens: int, do_sample=False):
        idx = tf.expand_dims(
            tf.convert_to_tensor(encode(prompt), dtype=tf.int64),
            axis=0)

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
            if do_sample:
                idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1, dtype=tf.int64) # (B, 1)
            else:
                idx_next = tf.argmax(probs, axis=-1, output_type=tf.int64)  # [1]
                idx_next = idx_next[:, tf.newaxis]                          # --> [1, 1]

            # concatenate to stream of integer indices
            idx = tf.concat([idx, idx_next], axis=1)    # (1,T) + (1,1) --> (1,T+1)
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
generated_text = decode(model.generate_text(prompt="good", max_new_tokens=500, do_sample=True))
print(generated_text)

