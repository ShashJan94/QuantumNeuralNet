import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

def create_quantum_circuit(n_qubits, parameters):
    qc = QuantumCircuit(n_qubits, n_qubits)  # Add classical bits for measurement
    for i in range(n_qubits):
        qc.rx(parameters[i], i)
        qc.ry(parameters[n_qubits + i], i)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Binary classification for simplicity
y = np.where(y > 0, 1, 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def encode_data(x):
    # Encode data into quantum circuit parameters
    n_qubits = len(x) // 2
    parameters = x[:2 * n_qubits]
    return parameters


class QNN:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.simulator = AerSimulator()
        self.overall_counts = Counter()

    def quantum_forward(self, x):
        parameters = encode_data(x)
        qc = create_quantum_circuit(self.n_qubits, parameters)
        transpiled_qc = transpile(qc, self.simulator)
        result = self.simulator.run(transpiled_qc, shots=1024).result()
        counts = result.get_counts()

        # Update overall counts for histogram
        self.overall_counts.update(counts)

        return counts

    def predict(self, X):
        y_pred = []
        for x in X:
            counts = self.quantum_forward(x)
            # Simple classification based on the measurement results
            if counts.get('00', 0) > counts.get('11', 0):
                y_pred.append(0)
            else:
                y_pred.append(1)
        return np.array(y_pred)

    def plot_histogram(self):
        plot_histogram(dict(self.overall_counts))
        plt.title('Overall QNN Measurement Results')
        plt.show()


# Initialize QNN with 2 qubits
qnn = QNN(n_qubits=2)

# Make predictions on the test set
y_pred_qnn = qnn.predict(X_test)

# Evaluate accuracy
accuracy_qnn = accuracy_score(y_test, y_pred_qnn)
print(f"QNN Accuracy: {accuracy_qnn}")

# Plot the overall histogram
qnn.plot_histogram()

# Plot confusion matrix
cm_qnn = confusion_matrix(y_test, y_pred_qnn)
disp_qnn = ConfusionMatrixDisplay(confusion_matrix=cm_qnn, display_labels=[0, 1])
disp_qnn.plot()
plt.title('QNN Confusion Matrix')
plt.show()

# Train a classical neural network on the training data
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred_mlp = mlp.predict(X_test)

# Evaluate accuracy
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Classical MLP Accuracy: {accuracy_mlp}")

# Plot confusion matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=[0, 1])
disp_mlp.plot()
plt.title('Classical MLP Confusion Matrix')
plt.show()

# Plot accuracies for comparison
labels = ['QNN', 'Classical MLP']
accuracies = [accuracy_qnn, accuracy_mlp]

plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison between QNN and Classical MLP')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()