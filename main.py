import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from semantic_statistics.src.misc import set_seed, generate_dataset
from semantic_statistics.src.model_td import ModelTD


class ToyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        q = self.activation(self.fc1(x))
        z = self.fc2(q)
        # Return logits for the loss function, and q for TD analysis
        return z, q

    def fit(self, X, y, epochs, lr=0.01):
        # We use raw logits here; PyTorch CrossEntropy handles the LogSoftmax
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for i in range(epochs):
            self.train()
            logits, _ = self.forward(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                # Calculate accuracy to verify it's actually learning
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean()
                print(f"Epoch {i + 1}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")


def main(input_dim=9, hidden_dim=16, n_bins=25):
    set_seed(42)

    # 1. Generate Random Inputs (Binary)
    X, y = generate_dataset(input_dim)
    print(f'X.shape = {X.shape}')

    # 2. Create and train the model
    model = ToyModel(input_dim, hidden_dim)  # Slightly more hidden units helps
    model.fit(X, y, epochs=300, lr=0.1)  # Lower LR for stability

    # 3. Thermodynamics
    system = ModelTD(X, y, model, n_bins=n_bins)
    E = system.get_E()
    S = system.get_S()
    beta = system.get_beta()
    T = system.get_T()
    C = system.get_C()

    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    plt.suptitle('Thermodynamic quantities for the NN')

    # Energy Distribution (histogram)
    ax[0, 0].hist(system.sampled_energies, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax[0, 0].set_title("Sampled Energy Distribution")
    ax[0, 0].set_xlabel("Energy ($p_{correct}$)")
    ax[0, 0].set_ylabel("Frequency")

    # Entropy plot
    ax[0, 1].plot(E, S, color='green', lw=2)
    ax[0, 1].set_title(r"Entropy $S(E)$")
    ax[0, 1].set_xlabel("Energy")

    # Beta plot
    ax[1, 0].plot(E, beta, color='orange', lw=2)
    ax[1, 0].set_title(r"Inverse Temperature $\beta(E)$")
    ax[1, 0].set_xlabel("Energy")

    # Specific heat scatter (plot)
    idx = np.argsort(T)
    ax[1, 1].scatter(T[idx], C[idx], color='red', lw=2)
    ax[1, 1].set_title("Specific Heat $C(T)$")
    ax[1, 1].set_xlabel("Temperature")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
