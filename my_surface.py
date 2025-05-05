
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from my_color import depolarizing_error
from itertools import combinations


global_simulator = AerSimulator()

x_stabilizer_matrix = [
    np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # X1 X2 X4 → 0 1 3
    np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # X2 X3 X5 → 1 2 4
    np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]),  # X4 X6 X7 X9 → 3 5 6 8
    np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]),  # X5 X7 X8 X10 → 4 6 7 9
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]),  # X9 X11 X12 → 8 10 11
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]),  # X10 X12 X13 → 9 11 12
]


def is_degenerate(error_vector, stabilizer_matrix = x_stabilizer_matrix):
    error_vector = np.array(error_vector)
    if np.all(error_vector):
        return False

    for r in range(len(stabilizer_matrix) + 1):
        for combo in combinations(stabilizer_matrix, r):
            total = np.zeros(len(error_vector), dtype=int)
            for vec in combo:
                total = (total + vec) % 2
            if np.array_equal(total, error_vector):
                return True
    return False

def generate_single_qubit_lut():
    NUM_QUBITS = 13
    NUM_STABILIZERS = 12

    # 0-indexed lists of which stabilizers each qubit affects
    x_stabilizers = { #qubit affetcted : stabilizer that returns -1
        0: [0],
        1: [0,1],
        2: [1],
        3: [0,2],
        4: [1,3],
        5: [2],
        6: [2,3],
        7: [3],
        8: [2,4],
        9: [3,5],
        10: [4],
        11: [4,5],
        12: [5],
    }

    z_stabilizers = {
        0: [6],
        1: [7],
        2: [8],
        3: [6,7],
        4: [7,8],
        5: [6,9],
        6: [7,10],
        7: [8,11],
        8: [9,10],
        9: [10,11],
        10: [9],
        11: [10],
        12: [11],
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


def apply_correction(logical: object, correction_ops: object) -> object:
    corrected = logical.copy()
    for qubit, pauli in correction_ops:
        if pauli in ['X', 'Y']:
            corrected[qubit] ^= 1

    return corrected


def surface_code(p=None):
    # 3 X-ancilla, 3 Z-ancilla, 13 data, plus their classical bits
    ar_x = QuantumRegister(6,  "ar_x")    # we have 6 X-stabilizers
    ar_z = QuantumRegister(6,  "ar_z")    # and 6 Z-stabilizers
    cl_x = ClassicalRegister(6, "cl_x")
    cl_z = ClassicalRegister(6, "cl_z")
    data = QuantumRegister(13, "data")
    cl_d = ClassicalRegister(13, "cl_data")
    qc = QuantumCircuit(ar_x, ar_z, cl_x, cl_z, data, cl_d)

    # --- X-stabilizers (measure Z-errors) ---
    #   s0 = X0 X1 X3
    #   s1 = X1 X2 X4
    #   s2 = X3 X5 X6 X8
    #   s3 = X4 X6 X7 X9
    #   s4 = X8 X10 X11
    #   s5 = X9 X11 X12
    x_stabs = [
      (0, [0,1,3]),
      (1, [1,2,4]),
      (2, [3,5,6,8]),
      (3, [4,6,7,9]),
      (4, [8,10,11]),
      (5, [9,11,12]),
    ]
    for anc, qubits in x_stabs:
        qc.h(ar_x[anc])
        for q in qubits:
            qc.cx(ar_x[anc], data[q])
        qc.h(ar_x[anc])
    qc.barrier()

    # inject noise
    depolarizing_error(qc, p, [data[i] for i in range(13)])
    qc.barrier()

    # measure X-syndrome
    for anc, qubits in x_stabs:
        qc.h(ar_x[anc])
        for q in qubits:
            qc.cx(ar_x[anc], data[q])
        qc.h(ar_x[anc])
        qc.measure(ar_x[anc], cl_x[anc])

    # --- Z-stabilizers (measure X-errors) ---
    #   s6 = Z0 Z3 Z5
    #   s7 = Z1 Z3 Z4 Z6
    #   s8 = Z2 Z4 Z7
    #   s9 = Z5 Z8 Z10
    #   s10= Z6 Z8 Z9 Z11
    #   s11= Z7 Z9 Z12
    z_stabs = [
      (0, [0,3,5]),
      (1, [1,3,4,6]),
      (2, [2,4,7]),
      (3, [5,8,10]),
      (4, [6,8,9,11]),
      (5, [7,9,12]),
    ]
    for anc, qubits in z_stabs:
        for q in qubits:
            qc.cx(data[q], ar_z[anc])
        qc.measure(ar_z[anc], cl_z[anc])

    qc.barrier()
    qc.measure(data, cl_d)
    return qc




def simulate_circuit(qc, LUT):
    simulator = global_simulator
    result = simulator.run(qc, shots=1).result()
    measurement = result.get_counts()
    #order of measurements is c_n .. c_0


    logical = list(measurement.keys())[0][:13][::-1]
    logical = [int(i) for i in logical]
    z_stab = list(measurement.keys())[0][14:20][::-1]
    x_stab = list(measurement.keys())[0][21::][::-1]
    for_Lut = x_stab + z_stab
    for_Lut = int(for_Lut, 2)
    if for_Lut in LUT:
        res = apply_correction(logical, LUT[for_Lut])
    else:
        res = logical
    # if is_degenerate(res):
    #     return (True, False)
    if_error = any(res)
    return (False, if_error)


def run_trials_for_p(p,n, LUT):
    total_errors = 0
    degen_count = 0
    for _ in range(n):
        is_degen, error = simulate_circuit(qc=surface_code(p), LUT=LUT)
        total_errors += error
        if is_degen:
            degen_count += 1
    degen_ratio = degen_count / n
    avg_error = total_errors / n
    return (p, avg_error, degen_ratio)



if __name__ == "__main__":

    n = 250  # number of trials per p
   # n = 2500
    p_values = np.arange(0.0001, 0.2, 0.005)
   # p_values = np.arange(0.01,1, 0.025)
   # p_values = np.linspace(0.001, 0.5, 40)


    LUT = generate_single_qubit_lut()


    results = []
    with ProcessPoolExecutor(max_workers = 13) as executor:
        futures = [executor.submit(run_trials_for_p, p, n, LUT) for p in p_values]
        for future in futures:
            results.append(future.result())

    results.sort()  # sort by p-value
    ps, qbers, degen_ratios = zip(*results)

    import numpy as np
    #
    #np.save('surface_nondegen_comp.npy', np.array(qbers))
   # np.save('degen_ratios_surface_2.npy', np.array(degen_ratios))

    # --- Display ---
    for p, qber, degen_ratio in results:
        print(f"p = {p:.5f} → QBER = {qber:.5f}", f"degen ratio: {degen_ratio:.5f}")

    # --- Plot ---
    plt.plot(ps, qbers, marker='o', ms=3)
    plt.xlabel('Depolarizing Probability (p)')
    plt.ylabel('(QBER)')
    plt.yscale("log")
    plt.grid(True)
    plt.title('Surface Code: QBER vs. Depolarizing Probability (degen)')
    plt.show()