import logging
import os
import matplotlib
import numpy as np
from enum import Enum

# Suppress warnings and set backend
logging.getLogger('qiskit_aer.noise.noise_model').setLevel(logging.ERROR)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- Import our custom modular packages ---
from shared_engine import run_simulation
from paper_symmetric import (STAB_A, build_code_a_circuit, get_symmetric_noise,
                             get_bound_symmetric_5_qubit)
from paper_asymmetric import (STAB_B, build_code_b_circuit, get_asymmetric_noise,
                              get_bound_asymmetric_9_qubit)
from paper_correlated import (build_code_c_circuit, get_correlated_noise,
                              get_bound_correlated_3_qubit)
from qiskit_aer.noise import NoiseModel
from paper_correlated_solution import build_paper_circuit, run_paper_simulation

def get_combined_noise(p_sym=0.0, p_asym=0.0, A=1, p_corr=0.0):
    master_noise = NoiseModel()
    if p_sym > 0: master_noise = get_symmetric_noise(p_sym, master_noise)
    if p_asym > 0: master_noise = get_asymmetric_noise(p_asym, A, master_noise)
    if p_corr > 0: master_noise = get_correlated_noise(p_corr, master_noise)
    return master_noise


# =====================================================================
# 1. API DEFINITIONS (The Enums)
# =====================================================================

class QCode(Enum):
    STEANE_5 = "steane_5"
    ASYM_9 = "asym_9"
    BURST_3 = "burst_3"
    BURST_SOLUTION = "burst_solution" # <-- Add this!

class QNoise(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    CORRELATED = "correlated"
    COMBINED = "combined"


# =====================================================================
# 2. THE CORE EXPERIMENT FUNCTION
# =====================================================================

def run_experiment(
        codes: list[QCode],
        noises: list[QNoise],
        p_start: float = 0.01,
        p_end: float = 0.20,
        p_steps: int = 20,
        shots: int = 1000,
        a_ratio: int = 3,
        save_plot: bool = True
):
    """
    Runs a highly parametrizable Quantum Error Correction experiment.
    """
    print(f"\n--- Starting Experiment ---")
    print(f"Codes:  {[c.name for c in codes]}")
    print(f"Noises: {[n.name for n in noises]}\n")

    p_values = np.linspace(p_start, p_end, p_steps)

    # Map the Enum objects to their specific functions and plot styles
    CODE_CONFIGS = {
        QCode.STEANE_5: {'build': build_code_a_circuit, 'method': 'stabilizer', 'stab': STAB_A,
                         'color': 'b', 'marker': 'X', 'label': 'Steane [[5,1,3]]'},
        QCode.ASYM_9: {'build': build_code_b_circuit, 'method': 'stabilizer', 'stab': STAB_B,
                       'color': 'r', 'marker': 's', 'label': 'Asymmetric [[9,1]]'},
        QCode.BURST_3: {'build': build_code_c_circuit, 'method': 'density_matrix', 'stab': None,
                        'color': 'g', 'marker': '^', 'label': 'Correlated Burst 3-Qubit'},
        # NEW ENTRY:
        QCode.BURST_SOLUTION: {'build': build_paper_circuit, 'method': 'custom', 'stab': None,
                               'color': 'm', 'marker': '*', 'label': 'Correlated Burst (Solution)'}
    }

    # Build only the requested circuits
    circuits = {code: CODE_CONFIGS[code]['build']() for code in codes}
    results = {noise: {code: [] for code in codes} for noise in noises}

    # --- SIMULATION LOOP ---
    for idx, p in enumerate(p_values):
        print(f"[{idx + 1}/{p_steps}] Simulating p = {p:.3f}...")

        # Generate active noise models dynamically
        active_noises = {}
        if QNoise.SYMMETRIC in noises: active_noises[QNoise.SYMMETRIC] = get_symmetric_noise(p)
        if QNoise.ASYMMETRIC in noises: active_noises[QNoise.ASYMMETRIC] = get_asymmetric_noise(p, a_ratio)
        if QNoise.CORRELATED in noises: active_noises[QNoise.CORRELATED] = get_correlated_noise(p)
        if QNoise.COMBINED in noises: active_noises[QNoise.COMBINED] = get_combined_noise(p, p, a_ratio, p)

        for n_enum in noises:
            for c_enum in codes:

                # --- NEW ROUTING LOGIC ---
                if c_enum == QCode.BURST_SOLUTION:
                    fail_rate = run_paper_simulation(
                        circuit=circuits[c_enum],
                        noise_model=active_noises[n_enum],
                        shots=shots
                    )
                else:
                    fail_rate = run_simulation(
                        circuit=circuits[c_enum],
                        noise_model=active_noises[n_enum],
                        shots=shots,
                        sim_method=CODE_CONFIGS[c_enum]['method'],
                        stabilizers=CODE_CONFIGS[c_enum]['stab']
                    )

                results[n_enum][c_enum].append(fail_rate)
    print("Simulations complete! Rendering plots...")

    # --- DYNAMIC PLOTTING ---
    num_plots = len(noises)
    # Update this line to enforce a minimum width!
    fig, axes = plt.subplots(1, num_plots, figsize=(max(7, 5.5 * num_plots), 6), squeeze=False)
    axes = axes.flatten()
    fig.suptitle('Quantum Error Correction Performance (Log-Log Scale)', fontsize=16)

    epsilon = 1e-6

    for i, n_enum in enumerate(noises):
        ax = axes[i]

        # 1. Plot simulated points
        for c_enum in codes:
            ax.loglog(p_values, np.maximum(results[n_enum][c_enum], epsilon),
                      marker=CODE_CONFIGS[c_enum]['marker'], color=CODE_CONFIGS[c_enum]['color'],
                      linestyle='None', alpha=0.6, label=f"{CODE_CONFIGS[c_enum]['label']} (Sim)")

        # 2. Plot Theoretical Bounds (if applicable to the current noise/code)
        if n_enum == QNoise.SYMMETRIC:
            if QCode.STEANE_5 in codes: ax.loglog(p_values, np.maximum(get_bound_symmetric_5_qubit(p_values), epsilon),
                                                  'b-', label='Steane Bound')
            if QCode.ASYM_9 in codes: ax.loglog(p_values,
                                                np.maximum(get_bound_asymmetric_9_qubit(p_values, A=1), epsilon), 'r-',
                                                label='Asym Bound (A=1)')

        elif n_enum == QNoise.ASYMMETRIC:
            if QCode.STEANE_5 in codes: ax.loglog(p_values, np.maximum(get_bound_symmetric_5_qubit(p_values), epsilon),
                                                  'b-', label='Steane Bound')
            if QCode.ASYM_9 in codes: ax.loglog(p_values,
                                                np.maximum(get_bound_asymmetric_9_qubit(p_values, a_ratio), epsilon),
                                                'r-', label=f'Asym Bound (A={a_ratio})')

        elif n_enum == QNoise.CORRELATED:
            if QCode.BURST_3 in codes: ax.loglog(p_values, np.maximum(get_bound_correlated_3_qubit(p_values), epsilon),
                                                 'g-', label='Burst Bound')

        # 3. Baseline & Styling
        ax.loglog(p_values, p_values, 'k--', label='Unencoded Baseline')
        ax.set_title(f"{n_enum.value.capitalize()} Noise")
        ax.set_xlabel('Physical Error Rate (p)')
        if i == 0: ax.set_ylabel('Logical Error Rate')
        ax.grid(True, which="both", linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- SAVING & SHOWING ---
    if save_plot:
        output_dir = "saved_plots"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        code_str = "_".join([c.value for c in codes])
        noise_str = "_".join([n.value[:4] for n in noises])
        file_path = os.path.join(output_dir, f"qec_{code_str}_{noise_str}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nPlot saved to: {file_path}")

    plt.show()


# =====================================================================
# 3. HOW TO USE THE API
# =====================================================================

# if __name__ == "__main__":
#     # Example 1: Run everything
#     # run_experiment(
#     #     codes=[QCode.STEANE_5, QCode.ASYM_9, QCode.BURST_3],
#     #     noises=[QNoise.SYMMETRIC, QNoise.ASYMMETRIC, QNoise.CORRELATED, QNoise.COMBINED]
#     # )
#
#     # Example 2: Just test the 9-qubit code against Asymmetric Noise
#     run_experiment(
#         codes=[QCode.ASYM_9],
#         noises=[QNoise.ASYMMETRIC],
#         a_ratio=10,  # Test a very extreme asymmetry bias
#         shots=1000,  # High precision
#         p_end=0.10  # Zoom in on the lower error bounds
#     )

if __name__ == "__main__":
    # The Ultimate Showdown: Standard QEC vs The Paper's Solution
    run_experiment(
        codes=[QCode.BURST_3, QCode.BURST_SOLUTION],
        noises=[QNoise.CORRELATED],
        shots=5000
    )