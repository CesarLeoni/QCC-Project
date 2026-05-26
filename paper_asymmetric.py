# paper_asymmetric.py
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit import transpile
from shared_engine import generate_memory_circuit, run_simulation

STAB_B = [
    'ZZIIIIIII',
    'IZZIIIIII',
    'IIIZZIIII',
    'IIIIZZIII',
    'IIIIIIZZI',
    'IIIIIIIZZ',
    'XXXXXXIII',
    'IIIXXXXXX'
]
def get_bound_asymmetric_9_qubit(p_array, A):
    p_x = p_array / (A + 2)
    p_y = p_array / (A + 2)
    p_z = A * p_array / (A + 2)
    return 36 * ((p_x ** 2) + (p_y ** 2) + (p_z ** 2))

def get_cwep_asymmetric_9_qubit(p_array, A):
    cwep = []
    for p in p_array:
        p_x = p / (A + 2)
        p_z = A * p / (A + 2)
        p_I = 1 - p
        p_0 = p_I ** 9
        p_1 = 9 * (p_I ** 8) * p
        p_2_correctable = 36 * (p_I ** 7) * (2 * (p_x * p_z))
        cwep.append(1 - (p_0 + p_1 + p_2_correctable))
    return np.array(cwep)

# def get_asymmetric_noise(p, A, noise_model=None):
#     noise = noise_model if noise_model is not None else NoiseModel()
#     p_x = p / (A + 2)
#     p_y = p / (A + 2)
#     p_z = A * p / (A + 2)
#     p_I = 1 - p
#     error = pauli_error([('X', p_x), ('Y', p_y), ('Z', p_z), ('I', p_I)])
#     noise.add_all_qubit_quantum_error(error, ['id', 'x', 'h'])
#     return noise


def get_asymmetric_noise(p, A, noise_model=None):
    noise = noise_model if noise_model is not None else NoiseModel()
    p_x = p / (A + 2)
    p_y = p / (A + 2)
    p_z = A * p / (A + 2)
    p_I = 1 - p
    error = pauli_error([('X', p_x), ('Y', p_y), ('Z', p_z), ('I', p_I)])

    # CHANGE THIS: Only apply noise to the 'id' (idle) gates!
    noise.add_all_qubit_quantum_error(error, ['id'])
    return noise

def build_code_b_circuit():
    return generate_memory_circuit(9, STAB_B)


# Add this temporary debug block to the bottom of paper_asymmetric.py
# if __name__ == "__main__":
#     circ = build_code_b_circuit()
#     # Turn noise OFF to see if we get a perfect syndrome
#     noise = NoiseModel()
#
#     simulator = AerSimulator(noise_model=noise)
#     compiled = transpile(circ, simulator)
#     counts = simulator.run(compiled, shots=1).result().get_counts()
#
#     print("Debug Output:", counts)
#     # If this prints {'00000000 00000000': 1}, your circuit is correct.
#     # If it prints something else, your circuit structure is wrong.



if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print("Testing Asymmetric Module Standalone...")

    # 1. Setup Parameters
    p_values = np.linspace(0.0001, 0.01, 20)
    shots = 10000
    A_ratio = 3
    epsilon = 1e-6

    circ = build_code_b_circuit()
    sim_results = []

    # 2. Run Simulation Loop
    for idx, p in enumerate(p_values):
        print(f"Simulating p = {p:.3f}...")
        noise = get_asymmetric_noise(p, A=A_ratio)
        fail_rate = run_simulation(
            circuit=circ,
            noise_model=noise,
            shots=shots,
            sim_method='stabilizer',
            stabilizers=STAB_B
        )
        sim_results.append(fail_rate)

    # 3. Calculate Analytical Math & Bounds
    math_exact = get_cwep_asymmetric_9_qubit(p_values, A=A_ratio)
    math_bound = get_bound_asymmetric_9_qubit(p_values, A=A_ratio)

    # 4. Plot the Results
    plt.figure(figsize=(8, 6))

    # Plot Sim, Math, Bound, and Baseline
    plt.loglog(p_values, np.maximum(sim_results, epsilon), 'rs', alpha=0.7, label='Simulated Results')
    plt.loglog(p_values, np.maximum(math_exact, epsilon), 'r-', alpha=0.5, label='Exact Math')
    plt.loglog(p_values, np.maximum(math_bound, epsilon), 'r--', label=f'Paper Bound (A={A_ratio})')
    plt.loglog(p_values, p_values, 'k:', label='Unencoded Baseline')

    # Styling
    plt.title(f'Chiani [[9,1]] Asymmetric Code Performance (A={A_ratio})', fontsize=14)
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()