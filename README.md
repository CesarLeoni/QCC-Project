
# Quantum Error Correction (QEC) Simulator

This project is a high-performance, modular simulation suite designed to validate Quantum Error Correction (QEC) codes against various noise models, including symmetric, asymmetric, and correlated burst noise. This simulator was developed to investigate the performance bounds of degenerate and nondegenerate codes as described in foundational QEC literature.

## 🛠 Project Structure
The repository is structured to separate physical circuit definitions from the experimental engine:

* **`main_comparison.py`**: The master orchestrator. Runs parallel experiments comparing multiple QEC codes across different noise environments.
* **`shared_engine.py`**: The core simulation engine. Contains generalized functions for syndrome measurement, lookup table generation, and noise-tolerant transpilation.
* **`paper_*.py`**: Individual modules for specific research papers. Each contains standalone testing blocks and mathematical bounds used to validate simulation results.
* **`paper_correlated_solution.py`**: A specialized implementation investigating degenerate codes for correlated burst noise, demonstrating how unique ancilla configurations can outperform standard repetition codes.

## 🚀 Getting Started

### Prerequisites
Ensure you have the required quantum computing libraries installed:
```bash
pip install -r requirements.txt

```

### Running Experiments

The project uses a modular API, allowing you to run localized tests or master comparisons.

**To run a standalone validation of a specific code:**

```bash
python paper_asymmetric.py

```

This will execute the code, run the simulation, and generate a matplotlib chart comparing your results against the paper's theoretical mathematical bounds.

**To run the master comparison:**
Configure your desired experiments in the `if __name__ == "__main__":` block of `main_comparison.py` and run:

```bash
python main_comparison.py

```

## 🔬 Methodology

This simulator validates QEC performance through **Code Capacity** simulations. By targeting noise specifically on idle (`id`) gates, we isolate the mathematical performance of the codes from hardware-level noise, allowing for direct verification against theoretical packing bounds.

### Key Features

* **Endian-Aware Decoding:** Implements custom syndrome mapping to account for Qiskit's little-endian measurement convention.
* **Dynamic Noise Injection:** Supports symmetric depolarizing, asymmetric bias ($A$-ratio), and fully correlated weight-3 burst noise models.
* **Threshold Analysis:** Automated generation of Log-Log plots to identify the "pseudo-threshold" where QEC codes begin to outperform the unencoded physical baseline.

## 📚 References

This simulator implements concepts from:

* [Simple Quantum Error Correcting Codes](https://arxiv.org/abs/quant-ph/9605021v1) - A.M. Steane
* [Short Codes for Quantum Channels with One Prevalent Pauli Error Type](https://arxiv.org/abs/2104.04365v1) - Marco Chiani, Lorenzo Valentini
* [Quantum error correction with degenerate codes for correlated noise](https://arxiv.org/abs/1007.3655v2) - Chiribella et al.

---

*Developed for research validation of quantum error correction performance.*