import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings

warnings.filterwarnings('ignore')

mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.values / 255.0
y = mnist.target.astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

print("MLP (scikit-learn)")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=20,
    batch_size=200,
    learning_rate_init=0.001,
    random_state=42,
    verbose=True
)

start_time = time.time()
mlp.fit(X_train, y_train)
mlp_train_time = time.time() - start_time
print(f"Время обучения MLP: {mlp_train_time:.2f} секунд")

print("\nПредсказание на тестовой выборке...")
y_pred_mlp = mlp.predict(X_test)

mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_precision = precision_score(y_test, y_pred_mlp, average='weighted')
mlp_recall = recall_score(y_test, y_pred_mlp, average='weighted')
mlp_f1 = f1_score(y_test, y_pred_mlp, average='weighted')

print("Результаты MLP:")
print(f"Accuracy: {mlp_accuracy:.4f}")
print(f"Precision: {mlp_precision:.4f}")
print(f"Recall: {mlp_recall:.4f}")
print(f"F1-score: {mlp_f1:.4f}")

print("CNN (LeNet) с PyTorch")

X_train_cnn = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
X_test_cnn = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train_cnn), y_train_tensor)
test_dataset = TensorDataset(torch.tensor(X_test_cnn), y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Используемое устройство: {device}")
print(f"Количество параметров: {sum(p.numel() for p in model.parameters())}")

num_epochs = 10
train_accuracies = []
test_accuracies = []

print("\nНачало обучения CNN...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total_train += target.size(0)
        correct_train += predicted.eq(target).sum().item()

    train_acc = 100. * correct_train / total_train
    train_accuracies.append(train_acc)

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total_test += target.size(0)
            correct_test += predicted.eq(target).sum().item()

    test_acc = 100. * correct_test / total_test
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

cnn_train_time = time.time() - start_time
print(f"\nВремя обучения CNN: {cnn_train_time:.2f} секунд")

print("\nПредсказание на тестовой выборке...")
model.eval()
y_pred_cnn = []
y_true_cnn = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        _, predicted = output.max(1)
        y_pred_cnn.extend(predicted.cpu().numpy())
        y_true_cnn.extend(target.numpy())

cnn_accuracy = accuracy_score(y_true_cnn, y_pred_cnn)
cnn_precision = precision_score(y_true_cnn, y_pred_cnn, average='weighted')
cnn_recall = recall_score(y_true_cnn, y_pred_cnn, average='weighted')
cnn_f1 = f1_score(y_true_cnn, y_pred_cnn, average='weighted')

print("Результаты CNN (LeNet):")
print(f"Accuracy: {cnn_accuracy:.4f}")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall: {cnn_recall:.4f}")
print(f"F1-score: {cnn_f1:.4f}")

print("\nТаблица сравнения:")
print("-" * 80)
print(f"{'Модель':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Время':<10}")
print("-" * 80)
print(
    f"{'MLP':<20} {mlp_accuracy:<12.4f} {mlp_precision:<12.4f} {mlp_recall:<12.4f} {mlp_f1:<12.4f} {mlp_train_time:<10.2f}")
print(
    f"{'CNN (LeNet)':<20} {cnn_accuracy:<12.4f} {cnn_precision:<12.4f} {cnn_recall:<12.4f} {cnn_f1:<12.4f} {cnn_train_time:<10.2f}")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Сравнение MLP и CNN на MNIST', fontsize=16)

models = ['MLP', 'CNN']
accuracies = [mlp_accuracy, cnn_accuracy]
axes[0, 0].bar(models, accuracies, color=['blue', 'green'])
axes[0, 0].set_ylim([0.95, 1.0])
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Точность')
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)

f1_scores = [mlp_f1, cnn_f1]
axes[0, 1].bar(models, f1_scores, color=['blue', 'green'])
axes[0, 1].set_ylim([0.95, 1.0])
axes[0, 1].set_ylabel('F1-score')
axes[0, 1].set_title('F1-score')
for i, v in enumerate(f1_scores):
    axes[0, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)

train_times = [mlp_train_time, cnn_train_time]
axes[0, 2].bar(models, train_times, color=['blue', 'green'])
axes[0, 2].set_ylabel('Время (сек)')
axes[0, 2].set_title('Время обучения')
for i, v in enumerate(train_times):
    axes[0, 2].text(i, v + 1, f'{v:.1f}с', ha='center', fontsize=10)

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)

disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
disp_mlp.plot(ax=axes[1, 0])
axes[1, 0].set_title('Матрица ошибок MLP')

disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn)
disp_cnn.plot(ax=axes[1, 1])
axes[1, 1].set_title('Матрица ошибок CNN')

axes[1, 2].plot(range(1, num_epochs + 1), train_accuracies, label='Train', marker='o')
axes[1, 2].plot(range(1, num_epochs + 1), test_accuracies, label='Test', marker='s')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Accuracy (%)')
axes[1, 2].set_title('Кривые обучения CNN')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Ошибки классификации CNN', fontsize=14)

misclassified_idx = [i for i in range(len(y_true_cnn)) if y_true_cnn[i] != y_pred_cnn[i]]

for i, ax in enumerate(axes.flat):
    if i < len(misclassified_idx):
        idx = misclassified_idx[i]
        img = X_test_cnn[idx].reshape(28, 28)
        true_label = y_true_cnn[idx]
        pred_label = y_pred_cnn[idx]

        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', fontsize=10)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Сверточные ядра CNN', fontsize=14)

weights_conv1 = model.conv1.weight.data.cpu().numpy()
for i in range(min(6, len(weights_conv1))):
    ax = axes[i // 3, i % 3]
    kernel = weights_conv1[i, 0]
    ax.imshow(kernel, cmap='coolwarm')
    ax.set_title(f'Ядро {i + 1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

