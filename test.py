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
    
    def forward(self, x):
        return self.fc(x)

# Инициализация модели в PyTorch
input_dim = 10
output_dim = 10
model_torch = SimpleModelTorch(input_dim, output_dim)

# Инициализация оптимизатора
optimizer_torch = optim.AdamW(model_torch.parameters(), lr=1e-3, weight_decay=1e-2, eps=1e-8)

# ------------------- Keras Model -------------------

# Пример модели Keras с явно заданными размерами
class SimpleModelKeras(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(SimpleModelKeras, self).__init__()
        self.fc = layers.Dense(output_dim, input_dim=input_dim)
    
    def call(self, inputs):
        return self.fc(inputs)

# Инициализация модели в Keras
model_keras = SimpleModelKeras(input_dim, output_dim)

# Обязательно строим модель перед установкой весов
model_keras.build((None, input_dim))  # Ожидаем, что входные данные будут иметь размерность (None, input_dim)

# Инициализация оптимизатора
optimizer_keras = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-2, epsilon=1e-8)

# ------------------- Инициализация одинаковых весов для обоих моделей -------------------

# Извлекаем веса из PyTorch
with torch.no_grad():
    torch_weights = model_torch.fc.weight.data
    torch_bias = model_torch.fc.bias.data

# Переводим их в numpy для Keras
keras_weights = torch_weights.numpy()
keras_bias = torch_bias.numpy()

# Применяем эти веса в модели Keras
model_keras.fc.set_weights([keras_weights.T, keras_bias])  # Транспонируем веса для Keras

# ------------------- Training Loop -------------------

# Данные для обеих моделей
data_torch = torch.randn(32, input_dim)  # PyTorch данные
target_torch = torch.randn(32, output_dim)  # Целевые данные

# Преобразуем данные PyTorch в формат numpy, чтобы использовать в Keras
data_keras = tf.convert_to_tensor(data_torch.numpy(), dtype=tf.float32)  # Keras данные
target_keras = tf.convert_to_tensor(target_torch.numpy(), dtype=tf.float32)  # Целевые данные Keras


# Шаг оптимизации в PyTorch
optimizer_torch.zero_grad()
output_torch = model_torch(data_torch)
loss_torch = torch.nn.functional.mse_loss(output_torch, target_torch)
loss_torch.backward()
optimizer_torch.step()

# Шаг оптимизации в Keras
with tf.GradientTape() as tape:
    output_keras = model_keras(data_keras)
    loss_keras = tf.reduce_mean(tf.square(output_keras - target_keras))
gradients_keras = tape.gradient(loss_keras, model_keras.trainable_variables)
optimizer_keras.apply_gradients(zip(gradients_keras, model_keras.trainable_variables))

# Печать значения loss для обеих моделей
print(f"PyTorch Loss: {loss_torch.item()}")
print(f"Keras Loss: {loss_keras.numpy()}")
