
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
import torchquantum as tq
import numpy as np

# Ensure device is set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define the classical CNN model to extract weights ===
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=5)
        self.fc1 = nn.Linear(12*4*4, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate and optionally load trained weights
cnn_model = CNNModel().to(device)
# If you have a saved model, load it like this:
# cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))

# Extract and flatten weights
numpy_weights = {}
nw_list = []
nw_list_normal = []

for name, param in cnn_model.state_dict().items():
    numpy_weights[name] = param.cpu().numpy()
for i in numpy_weights:
    nw_list.append(list(numpy_weights[i].flatten()))
for i in nw_list:
    for j in i:
        nw_list_normal.append(j)

n_qubit = int(np.ceil(np.log2(len(nw_list_normal))))

# === Define the LewHybridNN ===
class LewHybridNN(nn.Module):
    class QLayer(nn.Module):
        def __init__(self, n_blocks, shots=8192):
            super().__init__()
            self.n_wires = int(np.ceil(np.log2(len(nw_list_normal))))
            self.n_blocks = n_blocks
            self.shots = shots
            self.u3_layers = tq.QuantumModuleList()
            self.cu3_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.u3_layers.append(
                    tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires, has_params=True, trainable=True)
                )
                self.cu3_layers.append(
                    tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires, has_params=True, trainable=True, circular=True)
                )

        def forward(self):
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=next(self.parameters()).device)
            easy_scale_coeff = 2**(n_qubit - 1)
            gamma = 0.1
            beta = 0.8
            alpha = 0.3
            for k in range(self.n_blocks):
                self.u3_layers[k](qdev)
                self.cu3_layers[k](qdev)

            state_probs = qdev.get_states_1d().abs()[0].pow(2)[:len(nw_list_normal)]
            samples = torch.multinomial(state_probs, num_samples=self.shots, replacement=True)
            counts = torch.bincount(samples, minlength=len(state_probs)).float()
            measured_probs = counts / self.shots

            x = measured_probs.reshape(len(nw_list_normal), 1)
            x = (beta * torch.tanh(gamma * easy_scale_coeff * x)) ** alpha
            x = x - torch.mean(x)
            return x

    def __init__(self, shots=8192):
        super().__init__()
        self.QuantumNN = self.QLayer(n_blocks=4, shots=shots)
        self.fc = nn.Linear(len(nw_list_normal), 10)

    def forward(self, x):
        weights = self.QuantumNN()
        flat_x = x.view(x.size(0), -1)
        expanded_weights = weights.view(1, -1).expand(flat_x.size(0), -1)
        return self.fc(expanded_weights)

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

# Hyperparameters
step = 1e-4

def train_with_shots(shots, epochs=1):
    model = LewHybridNN(shots=shots).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=1e-5, eps=1e-6)

    acc_list = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        acc_list.append(acc)
        print(f"[Shots={shots}] Epoch {epoch+1}: Accuracy = {acc:.2f}%")

    return acc_list[-1]

# Sweep experiment
if __name__ == "__main__":
    shot_list = [8192, 16384, 32768, 65536]
    acc_results = []

    for shots in shot_list:
        acc = train_with_shots(shots, epochs=1)
        acc_results.append(acc)

    plt.plot(shot_list, acc_results, marker='o')
    plt.xlabel("Measurement Shots")
    plt.ylabel("Training Accuracy (%)")
    plt.title("Effect of Measurement Shots")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("shots_vs_accuracy.png", dpi=300)
    plt.show()
