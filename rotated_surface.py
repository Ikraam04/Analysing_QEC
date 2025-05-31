
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from my_color import depolarizing_error, apply_correction


"""
This is just 
"""



global_simulator = AerSimulator()

def generate_rotated_surface_code_LUT():
    NUM_QUBITS = 9
    NUM_STABILIZERS = 8

    # X-type stabilizers (0-3)
    x_stabilizers = {
        0:[0],  # q0 is in X₀
        1:[0, 1],  # q1 is in X₀, X₁
        2:[1],  # q2 is in X₁
        3:[0],  # q3 is in X₀
        4:[0, 2],  # q4 is in X₀, X₂
        5:[2],  # q5 is in X₂
        6:[3],  # q6 is in X₃
        7:[2, 3],  # q7 is in X₂, X₃
        8:[2],  # q8 is in X₂
    }

    # Z-type stabilizers (4-7)
    z_stabilizers = {
        0:[4],  # q0 is in Z₀
        1:[5],  # q1 is in Z₁
        2:[5],  # q2 is in Z₁
        3:[4, 6],  # q3 is in Z₀, Z₂
        4:[5, 6],  # q4 is in Z₁, Z₂
        5:[5, 7],  # q5 is in Z₁, Z₃
        6:[6],  # q6 is in Z₂
        7:[6],  # q7 is in Z₂
        8:[7],  # q8 is in Z₃
    }

    def syndrome_for_error(qubit, pauli):
        syndrome = [0] * NUM_STABILIZERS
        if pauli == 'X':
            for s in z_stabilizers.get(qubit, []):
                syndrome[s] = 1
        elif pauli == 'Z':
            for s in x_stabilizers.get(qubit, []):
                syndrome[s] = 1
        elif pauli == 'Y':
            for s in z_stabilizers.get(qubit, []):
                syndrome[s] = 1
            for s in x_stabilizers.get(qubit, []):
                syndrome[s] = 1
        return syndrome

    lut = {}

    for q in range(NUM_QUBITS):
        for pauli in ['X', 'Z', 'Y']:
            syn = syndrome_for_error(q, pauli)
            syndrome_str = int((''.join(str(b) for b in syn)),2)

            if syndrome_str not in lut:
                lut[syndrome_str] = [(q, pauli)]

    return lut

import numpy as np

x_stabilizer_matrix = [
    np.array([1, 1, 0, 1, 1, 0, 0, 0, 0]),  # X0 X1 X3 X4
    np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),  # X1 X2
    np.array([0, 0, 0, 0, 1, 1, 0, 1, 1]),  # X4 X5 X7 X8
    np.array([0, 0, 0, 0, 0, 0, 1, 1, 0]),  # X6 X7
]

from itertools import combinations

def is_degenerate(error_vector, stabilizer_matrix=x_stabilizer_matrix):
    error_vector = np.array(error_vector)
    if np.all(error_vector == 0):
        return False
    for r in range(len(stabilizer_matrix) + 1):
        for combo in combinations(stabilizer_matrix, r):
            total = np.zeros(len(error_vector), dtype=int)
            for vec in combo:
                total = (total + vec) % 2
            if np.array_equal(total, error_vector):
                return True
    return False

LUT_2 = generate_rotated_surface_code_LUT()
def rotated_surface(p=None):
    # 4 X‐ancilla, 4 Z‐ancilla, 9 data qubits + their classical bits
    ar_x = QuantumRegister(4, "ar_x")    # X‐syndrome
    ar_z = QuantumRegister(4, "ar_z")    # Z‐syndrome
    cl_x = ClassicalRegister(4, "cl_x")
    cl_z = ClassicalRegister(4, "cl_z")
    data = QuantumRegister(9, "data")
    cl_d = ClassicalRegister(9, "cl_data")
    qc = QuantumCircuit(ar_x, ar_z, cl_x, cl_z, data, cl_d)

    # --- X‐stabilizers (detect Z‐errors) ---
    # s0 = X1 X2
    # s1 = X0 X1 X3 X4
    # s2 = X4 X5 X7 X8
    # s3 = X6 X7
    x_stabs = [
        (0, [0,1,3,4]),
        (1, [1,2]),
        (2, [4, 5, 7, 8]),
        (3, [6, 7]),
    ]
    for anc, qubits in x_stabs:
        qc.h(ar_x[anc])
        for q in qubits:
            qc.cx(ar_x[anc], data[q])
        qc.h(ar_x[anc])
    qc.barrier()

    # inject depolarising noise on all data qubits
    depolarizing_error(qc, p, [data[i] for i in range(9)])
    qc.barrier()

    # measure X‐syndrome
    for anc, qubits in x_stabs:
        qc.h(ar_x[anc])
        for q in qubits:
            qc.cx(ar_x[anc], data[q])
        qc.h(ar_x[anc])
        qc.measure(ar_x[anc], cl_x[anc])

    # --- Z‐stabilizers (detect X‐errors) ---
    # s4 = Z0 Z3
    # s5 = Z1 Z2 Z4 Z5
    # s6 = Z3 Z4 Z6 Z7
    # s7 = Z5 Z8
    z_stabs = [
        (0, [0, 3]),
        (1, [1, 2, 4, 5]),
        (2, [3, 4, 6, 7]),
        (3, [5, 8]),
    ]
    for anc, qubits in z_stabs:
        for q in qubits:
            qc.cx(data[q], ar_z[anc])
        qc.measure(ar_z[anc], cl_z[anc])

    qc.barrier()
    qc.measure(data, cl_d)
    return qc


def simulate_rotated_surface_code(p, LUT):
    qc = rotated_surface(p)

    simulator = global_simulator
    result = simulator.run(qc, shots=1).result()
    measurement = result.get_counts()

    outcome_str = list(measurement.keys())[0]
    logical_str = outcome_str[:9][::-1]
    logical = [int(bit) for bit in logical_str]

    z_stab_str  = outcome_str[10:14][::-1]
    x_stab_str  = outcome_str[15:][::-1]


    for_Lut = x_stab_str + z_stab_str
    for_Lut = int(for_Lut, 2)

    if for_Lut in LUT:
        corrected_logical = apply_correction(logical, LUT[for_Lut])
    else:
        corrected_logical = logical

    """
    uncomment this to enable degeneracy!
    """
    # if is_degenerate(np.array(corrected_logical)):
    #     return (True,0)

    if_error = any(corrected_logical)  # returns 1 if error, else 0
    return (False, if_error)

def run_trials_for_p(p, n, LUT):
    total_errors = 0
    degen_count = 0
    for _ in range(n):
        degen, error = simulate_rotated_surface_code(p, LUT)
        total_errors += error
        if degen:
            degen_count += 1
    avg_degen = degen_count / n
    avg_error = total_errors / n
    return (p, avg_error, avg_degen)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from concurrent.futures import ProcessPoolExecutor
    n = 2500
    p_values = np.arange(0.0001, 0.2, 0.005)
    #p_values = np.arange(0.001, 0.4, 0.01)
   # p_values = np.linspace(0.001, 0.5, 40)

    LUT = generate_rotated_surface_code_LUT()


    results = []
    with ProcessPoolExecutor(max_workers=13) as executor:
        futures = [executor.submit(run_trials_for_p, p, n, LUT) for p in p_values]
        for future in futures:
            results.append(future.result())

    import numpy as np


    results.sort(key=lambda x: x[0])
    ps, qbers, degen_ratio = zip(*results)
    """
    save what you want, where you want, just make sure your consistent
    """
    np.save("rotated.npy", qbers)
    #np.save("rotated_nondegen_comp.npy", np.array(qbers))
    #np.save("degen_ratios_rotated.npy", np.array(degen_ratio))


    for p, qber, degen in results:
        print(f"p = {p:.5f} → QBER = {qber:.5f}, Degen ratio: {degen}")


    plt.plot(ps, qbers, marker='o', ms=3)
    plt.xlabel('Depolarizing Probability (p)')
    plt.ylabel('QBER')
    plt.yscale("log")
    plt.title('Rotated Surface Code: QBER vs. Depolarizing Probability')
    plt.grid(True)
    plt.show()
