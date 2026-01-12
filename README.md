
# 🧠 Neural Thermodynamics (NTD)

### *Exploring the Phase Transitions of Neural Logic*

This project implements a framework to evaluate the **Microcanonical Ensemble** of a neural network's activation space. By treating the hidden layer activations () as representative points in a phase space and the model's confidence () as a Hamiltonian energy function, we can derive classical thermodynamic quantities like **Entropy**, **Temperature**, and **Specific Heat**.

---

## 🔬 Core Concept

The central hypothesis is that a trained neural network behaves like a physical system undergoing a **phase transition**.

* **Disordered Phase (High ):** Before training (or during confusion), activations are a high-entropy "gas" with no clear structure.
* **Ordered Phase (Low ):** As the model learns, it "crystallizes." The activation space collapses into low-entropy manifolds where the energy (accuracy) is maximized.

This framework is specifically designed to detect **"Semantic Cooling"**—the phenomenon where a model becomes trapped in high-confidence, low-entropy states, which is a precursor to repetitive "stuttering" in Large Language Models.

---

## 🛠 Features

* **Deterministic Training:** Fully reproducible seeds across PyTorch, NumPy, and Python.
* **Activation Manifold Analysis:** Uses **Gaussian Kernel Density Estimation (KDE)** to estimate the density of states , correcting for non-uniform sampling in high-dimensional space.
* **Thermodynamic Solver:** Numerically derives the following functions:
* **Phase Space Volume ():** 
* **Entropy ():** 
* **Inverse Temperature ():** 
* **Specific Heat ():** 

---

## 📊 Interpreting the Results

When the program runs, it generates a four-quadrant diagnostic plot:

1. **Sampled Energy Distribution:** Shows how many states exist at each confidence level. A sharp peak at  indicates a "frozen" or perfectly trained model.
2. **Entropy :** Measures the logarithmic volume of the activation space. A sharp "kink" in this curve suggests a decision boundary.
3. **Inverse Temperature :** The derivative of entropy. High  means the model is "cold" and stable; low/negative  suggests a high-energy, unstable state.
4. **Specific Heat :** Look for spikes in this plot! Spikes in specific heat are the classic signature of a **phase transition** (e.g., the model shifting from "guessing" to "knowing").

---

## 🧪 Experimental Roadmap

* [x] **Phase 1:** Implement FFNN on binary classification.
* [x] **Phase 2:** Ensure determinism and importance sampling via KDE.
* [ ] **Phase 3:** Verify robustness and scalability.
* [ ] **Phase 4:** Test on GPT-2.

---

## 📄 License

MIT License. Feel free to use this for your own neural-physics research.
