import torch
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from semantic_statistics.src.misc import uniform_derivative, general_derivative


class ModelTD:
    def __init__(self, X, y, model, n_bins, gaussian_blur=1.):
        """
        :param X: data input
        :param y: one-hot data output
        :param model: Neural network with forward() method that returns (output proba, activations)
        :param n_bins: Number of bins for the plots
        :param gaussian_blur: Sigma param for gaussian blur on Omega
        """
        self.data_x = X
        self.data_y = y
        self.model = model
        self.n_bins = n_bins
        self.gaussian_blur = gaussian_blur

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

    @staticmethod
    def hamiltonian(p_correct):
        return p_correct

    def _sample_energies(self):
        self.model.eval()
        with torch.no_grad():
            logits, q_samples = self.model(self.data_x)
            probs = torch.softmax(logits, dim=1)
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
        e_min = self.sampled_energies.min()
        e_max = self.sampled_energies.max()
        self.energy = np.linspace(e_min, e_max, self.n_bins)

    def _evaluate_omega(self):
        """ Compute max-normalized omega
        Compute the volume of the phase space where H(q)<E.
        """
        # 1. Estimate rho(q) using KDE.
        # Note: In high dimensions, KDE is expensive. For N=4, it's fine.
        # Note: Standardizing q makes the algorithm more robust.
        # Transpose q to shape (features, samples) for scipy KDE
        normalized_q = self.sampled_q.T / np.std(self.sampled_q)
        kde = gaussian_kde(normalized_q)
        rho = kde.evaluate(normalized_q)

        # 2. Omega(E) = Sum [ 1(H < E) / rho(q) ]
        omega_values = []
        weights = 1.0 / (rho + 1e-9)
        for e in self.energy:
            mask = self.sampled_energies <= e
            vol_estimate = np.sum(weights[mask])
            omega_values.append(vol_estimate)
        self.omega = np.array(omega_values)

        # 2. Apply Gaussian Blur (Coarse-Graining)
        if self.gaussian_blur > 0:
            # We blur Omega before normalization to keep the integral smooth
            self.omega = gaussian_filter1d(self.omega, sigma=self.gaussian_blur)

        # 4. Max-normalize omega + Clip to avoid log(0)
        self.omega /= np.max(self.omega)
        self.omega = np.clip(self.omega, 1e-9, None)

    def _evaluate_entropy(self):
        """S = log Omega"""
        self.entropy = np.log(self.omega)

    def _evaluate_inv_temperature(self):
        """Beta = dS/dE"""
        self.inv_temperature = uniform_derivative(self.entropy, self.energy)

    def _evaluate_temperature(self):
        """T = 1/beta"""
        # Avoid division by zero in T
        self.temperature = 1.0 / (self.inv_temperature + 1e-9)

    def _evaluate_specific_heat(self):
        """C = dE/dT"""
        self.specific_heat = general_derivative(self.energy, self.temperature)

    # Getters
    def get_E(self): return self.energy
    def get_sampled_energies(self): return self.sampled_energies
    def get_S(self): return self.entropy, self.energy
    def get_beta(self): return self.inv_temperature, self.energy
    def get_T(self): return self.temperature, self.energy
    def get_C(self): return self.specific_heat, self.temperature
