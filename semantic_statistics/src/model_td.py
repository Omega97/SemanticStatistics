import torch
import numpy as np
from scipy.stats import gaussian_kde


class ModelTD:
    def __init__(self, X, y, model, n_bins=100):
        self.data_x = X
        self.data_y = y
        self.model = model
        self.n_bins = n_bins

        self.energy = None
        self.omega = None
        self.entropy = None
        self.inv_temperature = None
        self.temperature = None
        self.specific_heat = None

        # Compute the TD quantities
        self._sample_energies()
        self._set_energy()
        self._evaluate_omega()
        self._evaluate_entropy()
        self._evaluate_inv_temperature()
        self._evaluate_temperature()
        self._evaluate_specific_heat()

    def hamiltonian(self, p_correct):
        return p_correct

    def _sample_energies(self):
        self.model.eval()
        with torch.no_grad():
            # logits is 'z' from your forward method
            logits, q_samples = self.model(self.data_x)

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)

            # Now p_correct will be between 0 and 1
            i_indexes = range(len(self.data_y))
            j_indexes = self.data_y
            p_correct = probs[i_indexes, j_indexes].numpy()

            energies_raw = self.hamiltonian(p_correct)
            q_raw = q_samples.numpy()

        # Sort everything by energy for clean TD functions
        idx = np.argsort(energies_raw)
        self.sampled_energies = energies_raw[idx]
        self.sampled_q = q_raw[idx]

    def _set_energy(self):
        # Define the energy axis for our functions
        self.energy = np.linspace(self.sampled_energies.min(), self.sampled_energies.max(), self.n_bins)

    def _evaluate_omega(self):
        # 1. Estimate rho(q) using KDE.
        # Note: In high dimensions, KDE is expensive. For N=4, it's fine.
        # Transpose q to shape (features, samples) for scipy KDE
        kde = gaussian_kde(self.sampled_q.T)
        rho = kde.evaluate(self.sampled_q.T)

        # 2. Omega(E) = Sum [ 1(H < E) / rho(q) ]
        omega_values = []
        # Pre-calculate the weights
        weights = 1.0 / (rho + 1e-10)

        for e in self.energy:
            # Mask for samples where energy is <= e
            mask = self.sampled_energies <= e
            # Sum the corrected weights
            vol_estimate = np.sum(weights[mask])
            omega_values.append(vol_estimate)

        # Normalize omega so the max volume is 1.0 (relative measure)
        self.omega = np.array(omega_values)
        self.omega /= np.max(self.omega)
        # Clip to avoid log(0)
        self.omega = np.clip(self.omega, 1e-9, 1.0)

    def _evaluate_entropy(self):
        """S = log Omega"""
        self.entropy = np.log(self.omega)

    def _evaluate_inv_temperature(self):
        """Beta = dS/dE"""
        self.inv_temperature = np.gradient(self.entropy, self.energy)

    def _evaluate_temperature(self):
        """T = 1/beta"""
        # Avoid division by zero in T
        self.temperature = 1.0 / (self.inv_temperature + 1e-9)

    def _evaluate_specific_heat(self):
        """C = dE/dT"""
        self.specific_heat = np.gradient(self.energy, self.temperature)

    # Getters
    def get_E(self): return self.energy
    def get_S(self): return self.entropy
    def get_beta(self): return self.inv_temperature
    def get_T(self): return self.temperature
    def get_C(self): return self.specific_heat
