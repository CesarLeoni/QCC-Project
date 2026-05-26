from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


def generate_memory_circuit(num_data, stabilizers):
    num_ancilla = len(stabilizers)
    data = QuantumRegister(num_data, 'data')
    ancilla = QuantumRegister(num_ancilla, 'ancilla')
    synd1 = ClassicalRegister(num_ancilla, 'synd1')
    synd2 = ClassicalRegister(num_ancilla, 'synd2')

    # NEW: We must measure the actual data at the end to catch silent logical errors!
    data_meas = ClassicalRegister(num_data, 'data_meas')

    qc = QuantumCircuit(data, ancilla, synd1, synd2, data_meas)

    # 1. Project into codespace
    for i, stab in enumerate(stabilizers):
        qc.h(ancilla[i])
        for j, pauli in enumerate(stab):
            if pauli == 'X':
                qc.cx(ancilla[i], data[j])
            elif pauli == 'Z':
                qc.cz(ancilla[i], data[j])
            elif pauli == 'Y':
                qc.cy(ancilla[i], data[j])
        qc.h(ancilla[i])
        qc.measure(ancilla[i], synd1[i])

    qc.reset(ancilla)
    qc.barrier()

    # 2. Idle (Noise attacks here)
    for j in range(num_data):
        qc.id(data[j])
    qc.barrier()

    # 3. Read final syndrome
    for i, stab in enumerate(stabilizers):
        qc.h(ancilla[i])
        for j, pauli in enumerate(stab):
            if pauli == 'X':
                qc.cx(ancilla[i], data[j])
            elif pauli == 'Z':
                qc.cz(ancilla[i], data[j])
            elif pauli == 'Y':
                qc.cy(ancilla[i], data[j])
        qc.h(ancilla[i])
        qc.measure(ancilla[i], synd2[i])

    # 4. NEW: Measure data qubits
    qc.measure(data, data_meas)

    return qc


def build_lookup_table(stabilizers):
    lut = {}
    num_data = len(stabilizers[0])
    for q in range(num_data):
        for pauli in ['X', 'Y', 'Z']:
            synd = ""
            for stab in stabilizers:
                p_stab = stab[q]
                if pauli == 'I' or p_stab == 'I' or pauli == p_stab:
                    synd += "0"
                else:
                    synd += "1"
            synd = synd[::-1]
            if synd not in lut and synd != "0" * len(stabilizers):
                lut[synd] = (q, pauli)
    return lut


def run_simulation(circuit, noise_model, shots, sim_method, stabilizers=None):
    simulator = AerSimulator(noise_model=noise_model, method=sim_method)

    # CRITICAL FIX: optimization_level=0 stops Qiskit from deleting our id noise gates!
    compiled_circuit = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()

    num_cbits = circuit.num_clbits
    if num_cbits == 1:
        # Code C
        return counts.get('0', 0) / shots
    else:
        lut = build_lookup_table(stabilizers)
        logical_failures = 0

        for out_string, count in counts.items():
            parts = out_string.split()
            # Qiskit outputs little-endian: [Data] [Synd2] [Synd1]
            data_bits = parts[0]
            synd2 = parts[1]
            synd1 = parts[2]

            # BUG FIX: Reverse the data string so index 0 is actually Qubit 0!
            data_list = list(data_bits[::-1])
            #data_list = list(data_bits)
            error_synd = "".join(['0' if b1 == b2 else '1' for b1, b2 in zip(synd1, synd2)])

            # If the alarm went off, apply the software fix to the data string
            if error_synd != "0" * len(synd1):
                if error_synd not in lut:
                    logical_failures += count
                    continue
                else:
                    q, pauli = lut[error_synd]
                    if pauli in ['X', 'Y']:
                        data_list[q] = '0' if data_list[q] == '1' else '1'

            # FINAL CHECK: Did a silent logical error happen?
            # (If the fixed data parity is odd, the decoder failed us).
            final_parity = data_list.count('1') % 2
            if final_parity != 0:
                logical_failures += count

        return logical_failures / shots