import tensorflow as tf
from tensorflow import keras
from keras import layers

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 100
learning_rate = 1e-2
eval_iters = 200

tf.random.set_seed(1337)


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


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform((batch_size,), maxval=len(data) - block_size, dtype=tf.int64)
    x = tf.gather(data, ix[:, tf.newaxis] + tf.cast(tf.range(block_size), dtype=tf.int64))
    y = tf.gather(data, ix[:, tf.newaxis] + tf.cast(tf.range(1, block_size + 1), dtype=tf.int64))
    x, y = tf.convert_to_tensor(x, dtype=tf.int64), tf.convert_to_tensor(y, dtype=tf.int64)
    return x, y


@tf.function
def estimate_loss():
    out = {}
    model.trainable = False
    for split in ['train', 'val']:
        losses = tf.zeros(eval_iters, dtype=tf.float32)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses = tf.tensor_scatter_nd_add(losses, [[k]], [loss])
        out[split] = tf.reduce_mean(losses)
    model.trainable = True
    return out


class BigramLanguageModel(keras.Model):

    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = layers.Embedding(vocab_size, vocab_size)


    def call(self, idx, targets=None):

        # idx and targets are both (B=batch, T=time) tensor of integers
        logits = self.token_embedding_table(idx) # (B=batch, T=time, C=channel=vocab_size)
        # logits interprate the scores for the next character in the sequence

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in tf.range(max_new_tokens):
            # get the prediction by logits, loss ignored. 
            logits, loss = self(idx)    # forward to get logits = (B, T, C)
            # focus only on last time step
            logits = logits[:, -1, :]   # becomes (B, C)
            # apply softmax to get max probability
            probs = tf.nn.softmax(logits, axis=-1)  # (B, C)
            # get next char-index
            idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1, dtype=tf.int64) # (B, 1)
            # concatenate to stream of integer indices
            idx = tf.concat([idx, idx_next], axis=1)    # (B, T+1)
        return idx


def train_model(model: BigramLanguageModel):
    optimizer = tf.optimizers.Adam(learning_rate)

    for iter in tf.range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter.numpy()}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        with tf.GradientTape() as tape:
            # forward pass
            logits, loss = model(xb, yb)

        # backward pass
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))



model = BigramLanguageModel(vocab_size)
train_model(model)

# Generate text from the model
idx = tf.zeros((1, 1), dtype=tf.int64)  # (B, T)
generated_text = decode(model.generate(idx, max_new_tokens=500).numpy()[0])
print(generated_text)
