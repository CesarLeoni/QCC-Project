# paper_symmetric.py
import numpy as np
from scipy.special import comb
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from shared_engine import generate_memory_circuit, run_simulation

STAB_A = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']

def get_bound_symmetric_5_qubit(p_array):
    return 10 * (p_array ** 2)

def get_cwep_symmetric_5_qubit(p_array):
    cwep = []
    for p in p_array:
        p_0 = comb(5, 0) * (1 - p) ** 5
        p_1 = comb(5, 1) * (1 - p) ** 4 * p ** 1
        cwep.append(1 - (p_0 + p_1))
    return np.array(cwep)

def get_symmetric_noise(p, noise_model=None):
    noise = noise_model if noise_model is not None else NoiseModel()
    error = depolarizing_error(p, 1)
    noise.add_all_qubit_quantum_error(error, ['id'])
    return noise

def build_code_a_circuit():
    return generate_memory_circuit(5, STAB_A)

if __name__ == "__main__":
    circ = build_code_a_circuit()  # USE CODE A!
    noise = NoiseModel()

    simulator = AerSimulator(noise_model=noise)
    result = simulator.run(circ, shots=1).result()
    counts = result.get_counts()

    print("Debug Output:", counts)

# Standalone Test Runner
if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print("Testing Symmetric Module Standalone...")

    # 1. Setup Parameters
    p_values = np.linspace(0.01, 0.20, 15)
    shots = 10000
    epsilon = 1e-6

    circ = build_code_a_circuit()
    sim_results = []

    # 2. Run Simulation Loop
    for idx, p in enumerate(p_values):
        print(f"Simulating p = {p:.3f}...")
        noise = get_symmetric_noise(p)
        fail_rate = run_simulation(
            circuit=circ,
            noise_model=noise,
            shots=shots,
            sim_method='stabilizer',
            stabilizers=STAB_A
        )
        sim_results.append(fail_rate)

    # 3. Calculate Analytical Math & Bounds
    math_exact = get_cwep_symmetric_5_qubit(p_values)
    math_bound = get_bound_symmetric_5_qubit(p_values)

    # 4. Plot the Results
    plt.figure(figsize=(8, 6))

    # Plot Sim, Math, Bound, and Baseline
    plt.loglog(p_values, np.maximum(sim_results, epsilon), 'bX', alpha=0.7, label='Simulated Results')
    plt.loglog(p_values, np.maximum(math_exact, epsilon), 'b-', alpha=0.5, label='Exact Math')
    plt.loglog(p_values, np.maximum(math_bound, epsilon), 'b--', label='Paper Bound (10p^2)')
    plt.loglog(p_values, p_values, 'k:', label='Unencoded Baseline')

    # Styling
    plt.title('Steane [[5,1,3]] Symmetric Code Performance', fontsize=14)
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()