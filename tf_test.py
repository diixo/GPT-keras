from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf


batch_size = 32
seq_length = 32
embedding_dim = 256
dff = 256
num_heads = 4
num_layers = 4

# 1. Загрузка данных из текстового файла
file_path = 'input.txt'  # Путь к вашему текстовому файлу
with open(file_path, 'r', encoding='utf-8') as file:
    texts = file.read().splitlines()  # Разделяем на строки

# 2. Токенизация
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем паддинг токен в качестве EOS

# Токенизируем все строки в файле
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='tf', max_length=seq_length)


# 3. Создание модели с нуля
config = GPT2Config(
    vocab_size=tokenizer.vocab_size, 
    n_positions=seq_length,
    n_embd=embedding_dim, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_inner=dff
)
model = TFGPT2LMHeadModel(config)

# 4. Подготовка данных для обучения
# Для обучения модели необходимо подготовить входные и целевые данные.
# В нашем случае входные данные такие же, как и целевые (предсказание следующего токена).
inputs = encodings['input_ids']
labels = encodings['input_ids']

# Преобразуем их в tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

# Используем батчирование для обучения
batch_size = 2
dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)

# 5. Определение оптимизатора
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# 6. Определение функции потерь
def compute_loss(labels, logits):
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    return loss

# 7. Обучение модели с использованием model.fit()
# Для использования model.fit() в Keras, нужно определить модель, оптимизатор и метрики.
model.compile(optimizer=optimizer, loss=compute_loss)

# 8. Обучение модели
epochs = 3  # Количество эпох
model.fit(dataset, epochs=epochs)

# Сохранение модели после обучения
model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_model')
