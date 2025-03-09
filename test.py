import torch
import torch.optim as optim
import torch.nn as nn
from tensorflow import keras
from keras import layers, models
import tensorflow as tf


# ------------------- PyTorch Model -------------------
class SimpleModelTorch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModelTorch, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc(x)
        return self.relu(out)


input_dim = 10
output_dim = 10
model_torch = SimpleModelTorch(input_dim, output_dim)


optimizer_torch = optim.AdamW(model_torch.parameters(), lr=1e-3, weight_decay=1e-2, eps=1e-8)


# ------------------- Keras Model -------------------
class SimpleModelKeras(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(SimpleModelKeras, self).__init__()
        self.fc = layers.Dense(output_dim, input_dim=input_dim)
        self.relu = layers.ReLU()
    
    def call(self, inputs):
        out = self.fc(inputs)
        return self.relu(out)


model_keras = SimpleModelKeras(input_dim, output_dim)

model_keras.build((None, input_dim))  # (None, input_dim)


optimizer_keras = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-2, epsilon=1e-8)


# ------------------- Initialization equal weights for both models -------------------

# Extract weights from PyTorch
with torch.no_grad():
    torch_weights = model_torch.fc.weight.data
    torch_bias = model_torch.fc.bias.data


keras_weights = torch_weights.numpy()
keras_bias = torch_bias.numpy()


model_keras.fc.set_weights([keras_weights.T, keras_bias])

# ------------------- Training Loop -------------------

data_torch = torch.randn(32, input_dim)
target_torch = torch.randn(32, output_dim)

# Transform PyTorch into numpy, for using in Keras
data_keras = tf.convert_to_tensor(data_torch.numpy(), dtype=tf.float32)
target_keras = tf.convert_to_tensor(target_torch.numpy(), dtype=tf.float32)


# Step optimization in PyTorch
optimizer_torch.zero_grad()
output_torch = model_torch(data_torch)
loss_torch = torch.nn.functional.mse_loss(output_torch, target_torch)
loss_torch.backward()
optimizer_torch.step()

# Step optimization in Keras
with tf.GradientTape() as tape:
    output_keras = model_keras(data_keras)
    loss_keras = tf.reduce_mean(tf.square(output_keras - target_keras))
gradients_keras = tape.gradient(loss_keras, model_keras.trainable_variables)
optimizer_keras.apply_gradients(zip(gradients_keras, model_keras.trainable_variables))


print(f"PyTorch Loss: {loss_torch.item()}")
print(f"Keras Loss: {loss_keras.numpy()}")
