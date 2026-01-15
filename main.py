import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from semantic_statistics.src.misc import set_seed, generate_dataset, clip_to_nan
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


def main(input_dim=10, hidden_dim=40, gaussian_blur=.5,
         n_bins=50,
         epochs=1000, lr=0.1):
    set_seed(42)

    # 1. Generate Random Inputs (Binary)
    X, y = generate_dataset(input_dim)
    print(f'X.shape = {X.shape}')

    # 2. Create and train the model
    model = ToyModel(input_dim, hidden_dim)
    model.fit(X, y, epochs=epochs, lr=lr)

    # 3. Thermodynamics
    system = ModelTD(X, y, model, n_bins=n_bins, gaussian_blur=gaussian_blur)
    S, E = system.get_S()
    C, T = system.get_C()
    F, E = system.get_F()
    kapa, E = system.get_kapa()

    # Contain
    T = clip_to_nan(T, 0, 3)
    C = clip_to_nan(C, -3, 3)
    F = clip_to_nan(F, 0, 5)

    # 4. Plots
    fig, ax = plt.subplots(2, 3, figsize=(8, 6))
    plt.suptitle('Thermodynamic quantities for the NN')

    # Energy Distribution (histogram)
    a = ax[0, 0]
    a.hist(system.get_sampled_energies(), bins=n_bins, color='purple', alpha=0.7, edgecolor='black')
    a.set_title("Sampled Energy Distribution")
    a.set_xlabel("Energy ($p_{correct}$)")
    a.set_ylabel("Frequency")

    # Entropy plot
    a = ax[0, 1]
    a.plot(E, S, color='green', lw=2)
    a.set_title(r"Entropy $S(E)$")
    a.set_xlabel("Energy")

    # Beta plot
    # ax[1, 0].plot(E, beta, color='orange', lw=2)
    # ax[1, 0].set_title(r"Inverse Temperature $\beta(E)$")
    # ax[1, 0].set_xlabel("Energy")

    # Temperature
    a = ax[0, 2]
    a.plot(E, T, color='orange', lw=2)
    a.set_title(r"Temperature $T(E)$")
    a.set_xlabel("Energy")
    a.set_ylim(0, None)

    # Specific heat scatter (plot)
    a = ax[1, 0]
    # idx = np.argsort(T)
    a.scatter(T, C, color='red', lw=2)
    a.set_title("Specific Heat $C(T)$")
    a.set_xlabel("Temperature")

    # Free energy
    a = ax[1, 1]
    a.plot(E, F, color='blue', lw=2)
    a.set_title(r"Free Energy $F(E)$")
    a.set_xlabel("Energy")

    # Isothermal compressibility
    a = ax[1, 2]
    a.plot(E, kapa, color='purple', lw=2)
    a.set_title(r"Isoth. Compressibility $\kappa(E)$")
    a.set_xlabel("Energy")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
