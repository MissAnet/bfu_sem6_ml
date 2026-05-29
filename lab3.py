import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

print("=" * 60)
print("Лабораторная работа №3: LSTM генерация текста")
print("=" * 60)


def load_and_prepare_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    print(f"Размер текста: {len(text)} символов")

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Уникальных символов: {len(chars)}")

    return text, chars, char_to_idx, idx_to_char


def create_training_file():
    training_text = """Искусственный интеллект и машинное обучение развиваются очень быстро. 
Нейронные сети способны решать сложные задачи: распознавание изображений, 
обработка естественного языка, генерация текста и многое другое. 
LSTM (Long Short-Term Memory) - это особый тип рекуррентных нейронных сетей, 
который отлично подходит для работы с последовательностями данных. 
Он может запоминать информацию на длительное время и избегать проблемы затухающих градиентов. 
Текстовые файлы содержат последовательности символов, слов и предложений. 
Нейронная сеть может обучиться закономерностям в тексте и генерировать новые последовательности, 
которые будут похожи на оригинал. Это напоминает то, как люди учатся говорить и писать, 
подражая примерам. Генерация текста с помощью LSTM - увлекательная задача, 
которая показывает мощь современных методов глубокого обучения. 
Чем больше текста для обучения, тем лучше результаты генерации. 
Важно выбрать правильную архитектуру сети и параметры для достижения хороших результатов."""

    with open('training_text.txt', 'w', encoding='utf-8') as f:
        f.write(training_text)
    return 'training_text.txt'


def prepare_sequences(text, char_to_idx, seq_length=100):
    sequences = []
    next_chars = []

    for i in range(0, len(text) - seq_length, 1):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])

    print(f"Создано {len(sequences)} последовательностей")

    X = np.zeros((len(sequences), seq_length, len(char_to_idx)), dtype=np.bool_)
    y = np.zeros((len(sequences), len(char_to_idx)), dtype=np.bool_)

    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1

    return X, y


def create_model(vocab_size, seq_length):
    model = Sequential([
        LSTM(128, input_shape=(seq_length, vocab_size), return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.01),
        metrics=['accuracy']
    )

    return model


def generate_text(model, idx_to_char, char_to_idx, seed_text, length=500, temperature=0.5):
    generated = seed_text
    current_seq = seed_text[-100:] if len(seed_text) > 100 else seed_text

    for _ in range(length):
        x = np.zeros((1, 100, len(char_to_idx)))
        for t, char in enumerate(current_seq[-100:]):
            if char in char_to_idx:
                x[0, t, char_to_idx[char]] = 1

        preds = model.predict(x, verbose=0)[0]

        if temperature > 0:
            preds = np.log(preds + 1e-7) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            next_char = np.random.choice(len(char_to_idx), p=preds)
        else:
            next_char = np.argmax(preds)

        next_char = idx_to_char[next_char]
        generated += next_char
        current_seq += next_char

    return generated


print("\nПодготовка данных")

if not os.path.exists('training_text.txt'):
    file_path = create_training_file()
    print(f"Создан файл: {file_path}")
else:
    file_path = 'training_text.txt'

text, chars, char_to_idx, idx_to_char = load_and_prepare_text(file_path)

print("\nСоздание последовательностей")

SEQ_LENGTH = 100
X, y = prepare_sequences(text, char_to_idx, SEQ_LENGTH)

print("\nСоздание модели LSTM")

model = create_model(len(chars), SEQ_LENGTH)
model.summary()

print("\nОбучение модели")

history = model.fit(
    X, y,
    batch_size=128,
    epochs=50,
    verbose=1
)

print("\nГенерация текста")

seed_texts = [
    "Искусственный интеллект",
    "Нейронные сети",
    "LSTM",
    "Машинное обучение"
]

with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write("LSTM ГЕНЕРАЦИЯ ТЕКСТА - РЕЗУЛЬТАТЫ\n")
    f.write("=" * 80 + "\n\n")

    temperatures = [0.2, 0.5, 0.8, 1.0]

    for temp in temperatures:
        print(f"\nТемпература = {temp}")
        f.write(f"\nТЕМПЕРАТУРА = {temp}\n")
        f.write("=" * 80 + "\n\n")

        for seed in seed_texts:
            print(f"Seed: {seed[:30]}...")
            generated = generate_text(model, idx_to_char, char_to_idx, seed, length=300, temperature=temp)
            f.write(f"Seed: '{seed}'\n")
            f.write("-" * 40 + "\n")
            f.write(f"{generated}\n\n")
            print(f"Сгенерировано: {generated[:100]}...")

print("\nРезультаты сохранены в 'generated_text.txt'")

print("\nГенерация абракадабры")

short_seeds = ["А", "Б", "В", "Что", "Как", "Зачем"]

with open('abracadabra.txt', 'w', encoding='utf-8') as f:
    f.write("LSTM АБРАКАДАБРА\n")
    f.write("=" * 60 + "\n\n")

    for seed in short_seeds:
        print(f"\nSeed: '{seed}'")
        nonsense = generate_text(model, idx_to_char, char_to_idx, seed, length=150, temperature=1.2)
        print(f"Результат: {nonsense[:150]}...")
        f.write(f"Seed: '{seed}'\n")
        f.write(f"{nonsense}\n\n")

print("\nАбракадабра сохранена в 'abracadabra.txt'")

print("Файлы: training_text.txt, generated_text.txt, abracadabra.txt")
