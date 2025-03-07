
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


tf.random.set_seed(2081)


B, T, C = 4, 8, 2 # (Batches, Times, Channels)
x = tf.random.normal(shape=(B, T, C))
print(x.numpy())

# T=8 tokens are currently not talking to each other inside batch
# each token should communicate with previous tokes but do not next,
# because each next token is future token, that we should predict
# 6-th communicate only for 5-th, 4-th, 3-rd, 2-nd. 1-st etc
# avarage all preceding elements by communicate all channels from previous steps

################################################################################
'''          | col-0 col-1 col-N
(token)row-0 |
(token)row-1 |
(token)row-N |
'''


################################################################################
# version 1: simple history by set x[B, T] = mean_{i <= t} x[B, i]
xbow = np.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]                       # (t, C) = (rows, cols)
        # averaged by cols for all rows, and keep original cols-size
        mean = tf.reduce_mean(xprev, axis=0)    # (C,) = (cols,)
        # xbow keep original dimention, but averaged
        xbow[b, t] = mean                       # (b, t, C)
        #print(xprev.shape, mean.shape)

#print(xbow)


################################################################################
# version 2: Matrix multiplication as weighted aggregation
tril = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
summ = tf.reduce_sum(tril, axis=1, keepdims=True)   # (T, 1)
wei = tril / summ

# multiplication occurs along the last axis of the tensor x, and the result dimension will be (B, T, C)
xbow2 = tf.matmul(wei, x)  # (T,T) @ (B,T,C) --> (B,T,C)

print(np.isclose(xbow, xbow2, atol=1e-6))   # --> True


################################################################################
# version 3: use softmax
tril = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
wei = tf.zeros((T, T))
wei = tf.where(tril[:T, :T] == 0, float('-inf'), wei)
wei = tf.nn.softmax(wei, axis=-1)

xbow3 = tf.matmul(wei, x)  # (T,T) @ (B,T,C) --> (B,T,C)

print(np.isclose(xbow, xbow3, atol=1e-6))   # --> True


################################################################################
# version 4: implement self attention
B, T, C = 4, 8, 32  # (Batches, Times, Channels)
x = tf.random.normal(shape=(B, T, C))

# functionality of single Head perform the self-attention
head_size = 16

key = layers.Dense(head_size, use_bias=False)   # out_features=head_size
query = layers.Dense(head_size, use_bias=False) # out_features=head_size
value = layers.Dense(head_size, use_bias=False) # out_features=head_size

k = key(x)      # (B, T, head_size)
q = query(x)    # (B, T, head_size)
k_T = tf.transpose(k, perm=[0, 2, 1])   # (B, T, head_size) --> (B, head_size, T)
wei = tf.matmul(q, k_T)                 # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)


tril = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
#wei = tf.zeros((T, T))
wei = tf.where(tril[:T, :T] == 0, float('-inf'), wei)
wei = tf.nn.softmax(wei, axis=-1)
print(wei.numpy())

# x is private information of token
v = value(x)
xbow3 = tf.matmul(wei, v)  # (T, T) @ (B, T, head_size) --> (B, T, head_size)
print(xbow3.shape)
# print(xbow3.numpy())
