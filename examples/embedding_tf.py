import tensorflow as tf


class Embedding(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings = self.add_weight(
            shape=(input_dim, output_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2),
            trainable=True)
        self.shape = self.embeddings.shape
  
    def call(self, ids):
        # return row[.., output_dim]
        return tf.nn.embedding_lookup(self.embeddings, ids)


def test():

    vocab_sz = 1000
    context_sz = 32
    embedding = Embedding(input_dim=vocab_sz, output_dim=context_sz)
    print("Embedding.shape:", embedding.shape)


    logits = embedding(tf.constant([[1, 5, 10], [7, 2, 0]]))
    print(logits.shape) # (2, 3, 32)
    print(embedding.embeddings[1].numpy())  
    print(logits[0, 0].numpy()) # get row by [id, id]

    print(72 * "-")

    logits = embedding(tf.constant([1, 5, 10, 7, 2, 0]))
    print(logits.shape) # (6, 32)
    print(embedding.embeddings[1].numpy())  
    print(logits[0].numpy()) # get row by [id]

test()