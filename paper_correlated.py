# paper_correlated.py
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, pauli_error
from shared_engine import run_simulation
import numpy as np

def get_bound_correlated_3_qubit(p_array):
    return 3 * (p_array / 2)

def get_correlated_noise(p, noise_model=None):
    noise = noise_model if noise_model is not None else NoiseModel()
    error_corr = pauli_error([('XX', p / 2), ('ZZ', p / 2), ('II', 1 - p)])
    noise.add_all_qubit_quantum_error(error_corr, ['cx', 'cz', 'cy'])
    return noise

def build_code_c_circuit():
    qc = QuantumCircuit(3, 1)
    qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.barrier()
    qc.id(0); qc.id(1); qc.id(2)
    qc.barrier()
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.ccx(1, 2, 0)
    qc.measure(0, 0)
    return qc


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print("Testing Correlated Module Standalone...")

    # 1. Setup Parameters
    p_values = np.linspace(0.01, 0.20, 15)
    shots = 1000
    epsilon = 1e-6

    circ = build_code_c_circuit()
    sim_results = []

    # 2. Run Simulation Loop
    for idx, p in enumerate(p_values):
        print(f"Simulating p = {p:.3f}...")
        noise = get_correlated_noise(p)

        # Notice: No stabilizers passed, and method is density_matrix!
        fail_rate = run_simulation(
            circuit=circ,
            noise_model=noise,
            shots=shots,
            sim_method='density_matrix'
        )
        sim_results.append(fail_rate)

    # 3. Calculate Analytical Bound
    math_bound = get_bound_correlated_3_qubit(p_values)

    # 4. Plot the Results
    plt.figure(figsize=(8, 6))

    # Plot Sim, Bound, and Baseline
    plt.loglog(p_values, np.maximum(sim_results, epsilon), 'g^', alpha=0.7, label='Simulated Results')
    plt.loglog(p_values, np.maximum(math_bound, epsilon), 'g--', label='Paper Bound (Linear)')
    plt.loglog(p_values, p_values, 'k:', label='Unencoded Baseline')

    # Styling
    plt.title('Correlated Burst 3-Qubit Code Performance', fontsize=14)
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()