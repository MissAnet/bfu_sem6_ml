import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        self.last_input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.sigmoid(self.z)
        return self.output

    def backward(self, gradient_output, learning_rate):
        d_sigmoid = self.sigmoid_derivative(self.z)
        d_loss = gradient_output * d_sigmoid
        d_weights = self.last_input * d_loss
        d_bias = d_loss

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_loss * self.weights

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = [Neuron(layer_sizes[i]) for _ in range(layer_sizes[i + 1])]
            self.layers.append(layer)

    def forward(self, X):
        outputs = [X]
        for layer in self.layers:
            layer_outputs = []
            for neuron in layer:
                layer_outputs.append(neuron.forward(outputs[-1]))
            outputs.append(np.array(layer_outputs))
        self.outputs = outputs
        return outputs[-1]

    def backward(self, y_true, learning_rate):
        y_pred = self.outputs[-1][0]
        grad = 2 * (y_pred - y_true)

        for layer_idx in reversed(range(len(self.layers))):
            if layer_idx == len(self.layers) - 1:
                for neuron_idx, neuron in enumerate(self.layers[layer_idx]):
                    neuron.backward(grad, learning_rate)
                grad = np.zeros(len(self.layers[layer_idx - 1][0].last_input)) if layer_idx > 0 else None
                for neuron in self.layers[layer_idx]:
                    grad += neuron.backward(grad if isinstance(grad, np.ndarray) else 0, learning_rate)
            else:
                new_grad = np.zeros(len(self.layers[layer_idx][0].last_input))
                for neuron_idx, neuron in enumerate(self.layers[layer_idx]):
                    if neuron_idx < len(grad):
                        grad_to_prev = neuron.backward(grad[neuron_idx], learning_rate)
                        new_grad += grad_to_prev
                grad = new_grad

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                output = self.forward(X[i])
                loss = (output[0] - y[i]) ** 2
                total_loss += loss

                y_pred = self.outputs[-1][0]
                grad = 2 * (y_pred - y[i])

                for layer_idx in reversed(range(len(self.layers))):
                    if layer_idx == len(self.layers) - 1:
                        for neuron in self.layers[layer_idx]:
                            neuron.backward(grad, learning_rate)
                        if layer_idx > 0:
                            grad = np.zeros(len(self.layers[layer_idx - 1][0].last_input))
                            for neuron in self.layers[layer_idx]:
                                grad += neuron.backward(0, 0)
                    else:
                        new_grad = np.zeros(len(self.layers[layer_idx][0].last_input))
                        for neuron_idx, neuron in enumerate(self.layers[layer_idx]):
                            if neuron_idx < len(grad):
                                grad_to_prev = neuron.backward(grad[neuron_idx], learning_rate)
                                new_grad += grad_to_prev
                        grad = new_grad

            avg_loss = total_loss / len(X)
            losses.append(avg_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, X, threshold=0.5):
        predictions = []
        for x in X:
            output = self.forward(x)[0]
            predictions.append(1 if output >= threshold else 0)
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        for x in X:
            probas.append(self.forward(x)[0])
        return np.array(probas)

class SimpleNeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, y_true, learning_rate):
        grad = 2 * (self.activations[-1] - y_true)

        for i in range(len(self.weights) - 1, -1, -1):
            d_activation = grad * self.sigmoid_derivative(self.z_values[i])
            d_weights = np.outer(self.activations[i], d_activation)
            d_biases = d_activation

            self.weights[i] -= learning_rate * d_weights
            self.biases[i] -= learning_rate * d_biases

            grad = np.dot(d_activation, self.weights[i].T)

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                output = self.forward(X[i])
                loss = (output[0] - y[i]) ** 2
                total_loss += loss
                self.backward(np.array([y[i]]), learning_rate)

            avg_loss = total_loss / len(X)
            losses.append(avg_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, X, threshold=0.5):
        predictions = []
        for x in X:
            output = self.forward(x)[0]
            predictions.append(1 if output >= threshold else 0)
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        for x in X:
            probas.append(self.forward(x)[0])
        return np.array(probas)

iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

X_train_2d = X_train[:, [0, 2]]
X_test_2d = X_test[:, [0, 2]]


print("Один нейрон")

neuron = Neuron(2)
losses_neuron = []

for epoch in range(1000):
    total_loss = 0
    for i in range(len(X_train_2d)):
        output = neuron.forward(X_train_2d[i])
        loss = (output - y_train[i]) ** 2
        total_loss += loss
        neuron.backward(2 * (output - y_train[i]), learning_rate=0.1)
    if epoch % 100 == 0:
        avg_loss = total_loss / len(X_train_2d)
        losses_neuron.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

y_pred_train_neuron = [1 if neuron.forward(x) >= 0.5 else 0 for x in X_train_2d]
y_pred_test_neuron = [1 if neuron.forward(x) >= 0.5 else 0 for x in X_test_2d]

print("\nМетрики одного нейрона:")
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train_neuron):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test_neuron):.4f}")
print(f"Test Precision: {precision_score(y_test, y_pred_test_neuron):.4f}")
print(f"Test Recall: {recall_score(y_test, y_pred_test_neuron):.4f}")
print(f"Test F1: {f1_score(y_test, y_pred_test_neuron):.4f}")

print("Нейросеть: 2 слоя по 10 нейронов")

nn = SimpleNeuralNetwork([2, 10, 10, 1])
losses_nn = nn.train(X_train_2d, y_train, epochs=1000, learning_rate=0.1, verbose=True)

y_pred_train_nn = nn.predict(X_train_2d)
y_pred_test_nn = nn.predict(X_test_2d)

print("\nМетрики нейросети:")
print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train_nn):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test_nn):.4f}")
print(f"Test Precision: {precision_score(y_test, y_pred_test_nn):.4f}")
print(f"Test Recall: {recall_score(y_test, y_pred_test_nn):.4f}")
print(f"Test F1: {f1_score(y_test, y_pred_test_nn):.4f}")

def plot_decision_boundary(model, X, y, title, is_neuron=True):
    plt.figure(figsize=(10, 8))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = []
    for i in range(len(xx.ravel())):
        point = np.array([xx.ravel()[i], yy.ravel()[i]])
        if is_neuron:
            pred = model.forward(point)
        else:
            pred = model.forward(point)[0]
        Z.append(pred)
    Z = np.array(Z).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    colors = ['red' if label == 1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=50)

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 3')
    plt.title(title)
    plt.colorbar(label='Вероятность класса Setosa')
    plt.show()

plot_decision_boundary(neuron, X_test_2d, y_test,
                       "Разделяющая линия: Один нейрон", is_neuron=True)

# Визуализация для нейросети
plot_decision_boundary(nn, X_test_2d, y_test,
                       "Разделяющая линия: Нейросеть", is_neuron=False)

print("Сравнение метрик классификации")

metrics = {
    'Модель': ['Один нейрон', 'Нейросеть (2x10)'],
    'Test Accuracy': [
        accuracy_score(y_test, y_pred_test_neuron),
        accuracy_score(y_test, y_pred_test_nn)
    ],
    'Test Precision': [
        precision_score(y_test, y_pred_test_neuron),
        precision_score(y_test, y_pred_test_nn)
    ],
    'Test Recall': [
        recall_score(y_test, y_pred_test_neuron),
        recall_score(y_test, y_pred_test_nn)
    ],
    'Test F1': [
        f1_score(y_test, y_pred_test_neuron),
        f1_score(y_test, y_pred_test_nn)
    ]
}

print("\nТаблица сравнения:")
print("-" * 60)
print(f"{'Модель':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 60)
print(
    f"{'Один нейрон':<20} {metrics['Test Accuracy'][0]:<12.4f} {metrics['Test Precision'][0]:<12.4f} {metrics['Test Recall'][0]:<12.4f} {metrics['Test F1'][0]:<12.4f}")
print(
    f"{'Нейросеть (2x10)':<20} {metrics['Test Accuracy'][1]:<12.4f} {metrics['Test Precision'][1]:<12.4f} {metrics['Test Recall'][1]:<12.4f} {metrics['Test F1'][1]:<12.4f}")
print("-" * 60)

print("\nМатрицы ошибок:")
print("\nОдин нейрон:")
print(confusion_matrix(y_test, y_pred_test_neuron))
print("\nНейросеть:")
print(confusion_matrix(y_test, y_pred_test_nn))