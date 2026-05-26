import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


# =====================================================================
# 1. THE PAPER'S SPECIFIC CORRELATED NOISE MODEL
# =====================================================================
def get_paper_correlated_noise(p, noise_model=None):
    """
    The paper targets fully correlated weight-3 noise (Eq 23).
    When noise strikes, it hits all 3 qubits with the SAME Pauli error.
    """
    noise = noise_model if noise_model is not None else NoiseModel()

    # We distribute the probability 'p' evenly among XXX, YYY, and ZZZ errors
    error_corr = pauli_error([
        ('XXX', p / 3),
        ('YYY', p / 3),
        ('ZZZ', p / 3),
        ('III', 1 - p)
    ])

    # We only attack the idle gates (Phase 1 Code Capacity testing)
    noise.add_all_qubit_quantum_error(error_corr, ['id'])
    return noise


# =====================================================================
# 2. THE PAPER'S SPECIFIC CIRCUIT (The |0>|+> technique)
# =====================================================================
def build_paper_circuit():
    """
    Encodes 1 logical qubit into 3 physical qubits without entanglement!
    Data = |0>, Ancilla 0 = |0>, Ancilla 1 = |+>
    """
    data = QuantumRegister(1, 'data')
    ancilla = QuantumRegister(2, 'ancilla')
    synd = ClassicalRegister(2, 'synd')
    data_meas = ClassicalRegister(1, 'data_meas')

    qc = QuantumCircuit(data, ancilla, synd, data_meas)

    # 1. Initialization
    # Data and Ancilla 0 are naturally |0>. We set Ancilla 1 to |+>
    qc.h(ancilla[1])
    qc.barrier()

    # 2. Idle Memory Channel (Noise attacks here)
    qc.id(data[0])
    qc.id(ancilla[0])
    qc.id(ancilla[1])
    qc.barrier()

    # 3. Syndrome Measurement (The ingenious part)
    # Measure Ancilla 0 in the standard Z-basis
    qc.measure(ancilla[0], synd[0])

    # Measure Ancilla 1 in the X-basis (Hadamard then measure)
    qc.h(ancilla[1])
    qc.measure(ancilla[1], synd[1])
    qc.barrier()

    # 4. Final Data Measurement (To verify if we survived)
    qc.measure(data[0], data_meas[0])

    return qc


# =====================================================================
# 3. THE CUSTOM DECODER
# =====================================================================
def run_paper_simulation(circuit, noise_model, shots):
    """
    A custom decoder built specifically to read the |0>|+> syndrome.
    """
    simulator = AerSimulator(noise_model=noise_model)
    # optimization_level=0 ensures our id gates aren't deleted!
    compiled_circuit = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()

    logical_failures = 0

    for out_string, count in counts.items():
        # Qiskit output format: "data_meas synd" (e.g., "0 01")
        parts = out_string.split()
        data_bit = int(parts[0])
        synd = parts[1]  # 's1 s0' -> s1 is X-basis (anc1), s0 is Z-basis (anc0)

        # APPLY THE PAPER'S CORRECTIONS:
        if synd == '01':
            # XXX Error: ancilla 0 flipped to |1>, ancilla 1 stayed |+>
            data_bit = 1 - data_bit  # Fix the X flip

        elif synd == '10':
            # ZZZ Error: ancilla 0 stayed |0>, ancilla 1 flipped to |->
            pass  # A Z-error doesn't flip a Z-basis measurement, no action needed!

        elif synd == '11':
            # YYY Error: ancilla 0 flipped to |1>, ancilla 1 flipped to |->
            data_bit = 1 - data_bit  # Y acts as an X and Z, so we must fix the flip!

        # FINAL VERIFICATION: We encoded |0>, so the fixed data MUST be 0.
        if data_bit != 0:
            logical_failures += count

    return logical_failures / shots


# =====================================================================
# 4. STANDALONE TESTING & PLOTTING
# =====================================================================
if __name__ == "__main__":
    print("Testing The Paper's Novel 3-Qubit Code...")

    p_values = np.linspace(0.01, 0.20, 15)
    shots = 5000
    epsilon = 1e-6

    circ = build_paper_circuit()
    sim_results = []

    for idx, p in enumerate(p_values):
        print(f"Simulating p = {p:.3f}...")
        noise = get_paper_correlated_noise(p)
        fail_rate = run_paper_simulation(circ, noise, shots)
        sim_results.append(fail_rate)

    # The exact math bound for this code under this noise is ZERO!
    math_exact = np.zeros_like(p_values)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(p_values, np.maximum(sim_results, epsilon), 'm^', alpha=0.8, markersize=8, label='Simulated Results')
    plt.loglog(p_values, np.maximum(math_exact, epsilon), 'm-', alpha=0.5, label='Exact Math Bound (0 errors)')
    plt.loglog(p_values, p_values, 'k:', label='Unencoded Baseline')

    plt.title('Paper Solution: |0>|+> Code vs Correlated Noise', fontsize=14)
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()